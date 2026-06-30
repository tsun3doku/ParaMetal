#include "HeatModelRuntime.hpp"

#include "heat/HeatSystemPlayback.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "voronoi/VoronoiAdapters.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

HeatModelRuntime::HeatModelRuntime(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    CommandPool& cmdPool,
    float initialTemperatureValue)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      intrinsicMesh(intrinsicMesh),
      renderCommandPool(cmdPool),
      fixedTemperatureValue(initialTemperatureValue) {
    intrinsicVertexCount = intrinsicMesh.vertices.size();
    initialized = (intrinsicVertexCount > 0);
}

HeatModelRuntime::~HeatModelRuntime() {
    cleanup();
}

bool HeatModelRuntime::appendProduct(HeatProduct& product) {
    if (intrinsicVertexCount == 0 || intrinsicMesh.indices.size() < 3) {
        return false;
    }

    const size_t vertexCount = intrinsicVertexCount;
    std::vector<glm::vec3> positions(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        positions[i] = intrinsicMesh.vertices[i].position;
    }
    const std::vector<float> vertexAreas = computeVertexAreas(positions, intrinsicMesh.indices);

    std::vector<heat::SurfacePoint> surfacePoints(vertexCount);
    for (size_t i = 0; i < vertexCount; ++i) {
        surfacePoints[i].position    = positions[i];
        surfacePoints[i].temperature = fixedTemperatureValue;
        surfacePoints[i].normal      = intrinsicMesh.vertices[i].normal;
        surfacePoints[i].area        = (i < vertexAreas.size()) ? vertexAreas[i] : 0.0f;
        surfacePoints[i].color       = glm::vec4(1.0f);
    }

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    VkBuffer surfBuf = VK_NULL_HANDLE;
    VkDeviceSize surfOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, renderCommandPool,
            surfacePoints.data(), sizeof(heat::SurfacePoint) * vertexCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
            storageAlignment, surfBuf, surfOffset) != VK_SUCCESS) {
        return false;
    }

    std::vector<glm::vec4> zeroGradients(vertexCount, glm::vec4(0.0f));
    VkBuffer gradBuf = VK_NULL_HANDLE;
    VkDeviceSize gradOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, renderCommandPool,
            zeroGradients.data(), sizeof(glm::vec4) * vertexCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment, gradBuf, gradOffset) != VK_SUCCESS) {
        memoryAllocator.free(surfBuf, surfOffset);
        return false;
    }

    product.modelSurfaceBuffers.push_back(surfBuf);
    product.modelSurfaceBufferOffsets.push_back(surfOffset);
    product.modelSurfacePointCounts.push_back(static_cast<uint32_t>(vertexCount));
    product.modelSurfaceGradientBuffers.push_back(gradBuf);
    product.modelSurfaceGradientBufferOffsets.push_back(gradOffset);

    return true;
}

void HeatModelRuntime::setBoundaryCondition(uint32_t bc) {
    boundaryCondition = bc;
}

void HeatModelRuntime::setFixedTemperatureValue(float temperature) {
    fixedTemperatureValue = temperature;
}

void HeatModelRuntime::setMaterialProperties(float density, float specificHeat, float conductivity) {
    this->density = density;
    this->specificHeat = specificHeat;
    this->conductivity = conductivity;
}

void HeatModelRuntime::cleanup() {
    if (playback) {
        playback->cleanup();
        playback.reset();
    }

    freeBuffer(memoryAllocator, materialBuffer, materialBufferOffset);

    gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    gmlsSurfaceStencilBufferOffset = 0;
    gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    gmlsSurfaceWeightBufferOffset = 0;
    gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    gmlsSurfaceGradientWeightBufferOffset = 0;

    surfaceComputeSetA = VK_NULL_HANDLE;
    surfaceComputeSetB = VK_NULL_HANDLE;
    surfaceGradientComputeSetA = VK_NULL_HANDLE;
    surfaceGradientComputeSetB = VK_NULL_HANDLE;
    surfaceHistoryComputeSetA = VK_NULL_HANDLE;
    surfaceHistoryComputeSetB = VK_NULL_HANDLE;
    surfaceGradientHistorySetA = VK_NULL_HANDLE;
    surfaceGradientHistorySetB = VK_NULL_HANDLE;

    cleanupSimulationBuffers();
    initialized = false;
}

void HeatModelRuntime::setHistoryBuffer(VkBuffer buffer, VkDeviceSize offset, uint32_t frameCapacity) {
    historyBuffer = buffer;
    historyBufferOffset = offset;
    historyBufferFrameCapacity = frameCapacity;
}

void HeatModelRuntime::initializePlayback(VulkanDevice& device, MemoryAllocator& allocator, uint32_t frameCapacity) {
    if (simNodeCount == 0 || frameCapacity == 0) return;
    ensureSimulationBuffers(simNodeCount);
    if (!playback) {
        playback = std::make_unique<HeatSystemPlayback>();
    }
    if (!playback->isValid() || playback->getFrameCapacity() != frameCapacity || playback->getNodeCount() != simNodeCount) {
        playback->initialize(device, allocator, simNodeCount, frameCapacity);
    }
    if (playback->isValid()) {
        setHistoryBuffer(playback->getHistoryBuffer(), playback->getHistoryBufferOffset(), frameCapacity);
    }
}

void HeatModelRuntime::setGMLSSurfaceWeights(
    VkBuffer stencilBuffer,
    VkDeviceSize stencilBufferOffset,
    VkBuffer valueWeightBuffer,
    VkDeviceSize valueWeightBufferOffset,
    size_t valueWeightCount,
    VkBuffer gradientWeightBuffer,
    VkDeviceSize gradientWeightBufferOffset,
    size_t gradientWeightCount) {
    gmlsSurfaceStencilBuffer = stencilBuffer;
    gmlsSurfaceStencilBufferOffset = stencilBufferOffset;
    gmlsSurfaceWeightBuffer = valueWeightBuffer;
    gmlsSurfaceWeightBufferOffset = valueWeightBufferOffset;
    gmlsSurfaceWeightCount = valueWeightCount;
    gmlsSurfaceGradientWeightBuffer = gradientWeightBuffer;
    gmlsSurfaceGradientWeightBufferOffset = gradientWeightBufferOffset;
    gmlsSurfaceGradientWeightCount = gradientWeightCount;
}

bool HeatModelRuntime::updateAllDescriptors(
    VkBuffer surfaceBuffer,
    VkDeviceSize surfaceBufferOffset,
    VkBuffer surfaceGradientBuffer,
    VkDeviceSize surfaceGradientBufferOffset,
    VkDescriptorSetLayout surfaceLayout,
    VkDescriptorSetLayout gradientLayout,
    VkDescriptorPool surfacePool,
    VkDescriptorSetLayout voronoiLayout,
    VkDescriptorPool voronoiPool,
    VkBuffer playbackBuffer,
    VkDeviceSize playbackBufferOffset,
    VkBuffer historyBuffer,
    VkDeviceSize historyBufferOffset,
    bool forceReallocate) {
    const size_t intrinsicVertexCount = getIntrinsicVertexCount();
    if (intrinsicVertexCount == 0 ||
        gmlsSurfaceStencilBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceWeightBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceGradientWeightBuffer == VK_NULL_HANDLE ||
        surfaceBuffer == VK_NULL_HANDLE ||
        surfaceGradientBuffer == VK_NULL_HANDLE ||
        playbackBuffer == VK_NULL_HANDLE ||
        historyBuffer == VK_NULL_HANDLE ||
        simNodeBuffer == VK_NULL_HANDLE ||
        simGMLSInterfaceBuffer == VK_NULL_HANDLE ||
        materialBuffer == VK_NULL_HANDLE ||
        tempBufferA == VK_NULL_HANDLE ||
        tempBufferB == VK_NULL_HANDLE ||
        contactAccumulatorBuffer == VK_NULL_HANDLE ||
        simNodeCount == 0 ||
        simGMLSInterfaceCount == 0 ||
        historyBufferFrameCapacity == 0) {
        return false;
    }

    if (forceReallocate) {
        surfaceComputeSetA = VK_NULL_HANDLE;
        surfaceComputeSetB = VK_NULL_HANDLE;
        surfaceGradientComputeSetA = VK_NULL_HANDLE;
        surfaceGradientComputeSetB = VK_NULL_HANDLE;
        surfaceHistoryComputeSetA = VK_NULL_HANDLE;
        surfaceHistoryComputeSetB = VK_NULL_HANDLE;
        surfaceGradientHistorySetA = VK_NULL_HANDLE;
        surfaceGradientHistorySetB = VK_NULL_HANDLE;
        voronoiDescriptorSetA = VK_NULL_HANDLE;
        voronoiDescriptorSetB = VK_NULL_HANDLE;
    }

    if (surfaceComputeSetA == VK_NULL_HANDLE || surfaceComputeSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> layouts = { surfaceLayout, surfaceLayout };
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, surfacePool, 2, layouts.data()};
        VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, sets) == VK_SUCCESS) {
            surfaceComputeSetA = sets[0];
            surfaceComputeSetB = sets[1];
        }
    }

    if (surfaceGradientComputeSetA == VK_NULL_HANDLE || surfaceGradientComputeSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> gradLayouts = { gradientLayout, gradientLayout };
        VkDescriptorSetAllocateInfo gradAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, surfacePool, 2, gradLayouts.data()};
        VkDescriptorSet gradSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &gradAllocInfo, gradSets) == VK_SUCCESS) {
            surfaceGradientComputeSetA = gradSets[0];
            surfaceGradientComputeSetB = gradSets[1];
        }
    }

    if (surfaceHistoryComputeSetA == VK_NULL_HANDLE || surfaceHistoryComputeSetB == VK_NULL_HANDLE ||
        surfaceGradientHistorySetA == VK_NULL_HANDLE || surfaceGradientHistorySetB == VK_NULL_HANDLE) {

        std::vector<VkDescriptorSetLayout> historyLayouts = { surfaceLayout, surfaceLayout, gradientLayout, gradientLayout };
        VkDescriptorSetAllocateInfo historyAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, surfacePool, 4, historyLayouts.data()};
        VkDescriptorSet historySets[4] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &historyAllocInfo, historySets) == VK_SUCCESS) {
            surfaceHistoryComputeSetA = historySets[0];
            surfaceHistoryComputeSetB = historySets[1];
            surfaceGradientHistorySetA = historySets[2];
            surfaceGradientHistorySetB = historySets[3];
        }
    }

    if (voronoiDescriptorSetA == VK_NULL_HANDLE || voronoiDescriptorSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> vLayouts = { voronoiLayout, voronoiLayout };
        VkDescriptorSetAllocateInfo vAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, voronoiPool, 2, vLayouts.data()};
        VkDescriptorSet vSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &vAllocInfo, vSets) == VK_SUCCESS) {
            voronoiDescriptorSetA = vSets[0];
            voronoiDescriptorSetB = vSets[1];
        }
    }

    if (surfaceComputeSetA == VK_NULL_HANDLE ||
        surfaceGradientComputeSetA == VK_NULL_HANDLE ||
        surfaceHistoryComputeSetA == VK_NULL_HANDLE ||
        surfaceGradientHistorySetA == VK_NULL_HANDLE ||
        voronoiDescriptorSetA == VK_NULL_HANDLE) return false;

    // Common info
    VkDescriptorBufferInfo surfaceInfo{ surfaceBuffer, surfaceBufferOffset, sizeof(heat::SurfacePoint) * intrinsicVertexCount };
    VkDescriptorBufferInfo gradientInfo{ surfaceGradientBuffer, surfaceGradientBufferOffset, sizeof(glm::vec4) * intrinsicVertexCount };
    VkDescriptorBufferInfo playbackInfo{ playbackBuffer, playbackBufferOffset, sizeof(heat::SimPlaybackUniform) };
    VkDescriptorBufferInfo historyInfo{ historyBuffer, historyBufferOffset, static_cast<VkDeviceSize>(historyBufferFrameCapacity) * simNodeCount * sizeof(float) };
    VkDescriptorBufferInfo gmlsStencilInfo{ gmlsSurfaceStencilBuffer, gmlsSurfaceStencilBufferOffset, intrinsicVertexCount * sizeof(voronoi::GMLSSurfaceStencil) };
    VkDescriptorBufferInfo gmlsValueWeightInfo{ gmlsSurfaceWeightBuffer, gmlsSurfaceWeightBufferOffset, gmlsSurfaceWeightCount * sizeof(voronoi::GMLSSurfaceWeight) };
    VkDescriptorBufferInfo gmlsGradientWeightInfo{ gmlsSurfaceGradientWeightBuffer, gmlsSurfaceGradientWeightBufferOffset, gmlsSurfaceGradientWeightCount * sizeof(voronoi::GMLSSurfaceGradientWeight) };
    
    VkDescriptorBufferInfo vNodeInfo{simNodeBuffer, simNodeOffset, simNodeCount * sizeof(voronoi::Node)};
    VkDescriptorBufferInfo vGmlsInfo{simGMLSInterfaceBuffer, simGMLSInterfaceOffset, simGMLSInterfaceCount * sizeof(voronoi::GMLSInterface)};
    VkDescriptorBufferInfo vMatInfo{materialBuffer, materialBufferOffset, simNodeCount * sizeof(heat::MaterialNode)};
    const VkDescriptorSet sTempSets[2] = { surfaceComputeSetA, surfaceComputeSetB };
    const VkDescriptorSet sGradSets[2] = { surfaceGradientComputeSetA, surfaceGradientComputeSetB };
    const VkDescriptorSet vSets[2] = { voronoiDescriptorSetA, voronoiDescriptorSetB };
    const VkBuffer tempBuffers[2] = { tempBufferA, tempBufferB };
    const VkDeviceSize tempOffsets[2] = { tempBufferAOffset, tempBufferBOffset };

    for (uint32_t pass = 0; pass < 2; ++pass) {
        VkDescriptorBufferInfo nodeTempInfo{ tempBuffers[pass], tempOffsets[pass], simNodeCount * sizeof(float) };
        VkDescriptorBufferInfo nodeNextTempInfo{ tempBuffers[1-pass], tempOffsets[1-pass], simNodeCount * sizeof(float) };
        VkDescriptorBufferInfo fluxInfo{ contactAccumulatorBuffer, contactAccumulatorBufferOffset, simNodeCount * sizeof(float) * 2 };

        // Surface Temperature Updates
        std::array<VkWriteDescriptorSet, 4> sTempWrites{};
        sTempWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        sTempWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
        sTempWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
        sTempWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsValueWeightInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), 4, sTempWrites.data(), 0, nullptr);

        // Surface Gradient Updates
        std::array<VkWriteDescriptorSet, 5> sGradWrites{};
        sGradWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        sGradWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
        sGradWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gradientInfo, nullptr};
        sGradWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
        sGradWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsGradientWeightInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), 5, sGradWrites.data(), 0, nullptr);

        // Diffusion
        std::array<VkWriteDescriptorSet, 7> vWrites{};
        vWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vNodeInfo, nullptr};
        vWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vGmlsInfo, nullptr};
        vWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vMatInfo, nullptr};
        vWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &playbackInfo, nullptr};
        vWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        vWrites[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeNextTempInfo, nullptr};
        vWrites[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fluxInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(vWrites.size()), vWrites.data(), 0, nullptr);
    }

    {
        VkDescriptorBufferInfo historyFrameInfo{ historyBuffer, historyBufferOffset, static_cast<VkDeviceSize>(historyBufferFrameCapacity) * simNodeCount * sizeof(float) };

        const VkDescriptorSet historySets[2] = { surfaceHistoryComputeSetA, surfaceHistoryComputeSetB };
        const VkDescriptorSet gradHistorySets[2] = { surfaceGradientHistorySetA, surfaceGradientHistorySetB };
        for (uint32_t pass = 0; pass < 2; ++pass) {
            std::array<VkWriteDescriptorSet, 4> hTempWrites{};
            hTempWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, historySets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &historyFrameInfo, nullptr};
            hTempWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, historySets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
            hTempWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, historySets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
            hTempWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, historySets[pass], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsValueWeightInfo, nullptr};
            vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(hTempWrites.size()), hTempWrites.data(), 0, nullptr);

            std::array<VkWriteDescriptorSet, 5> hGradWrites{};
            hGradWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gradHistorySets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &historyFrameInfo, nullptr};
            hGradWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gradHistorySets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
            hGradWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gradHistorySets[pass], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gradientInfo, nullptr};
            hGradWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gradHistorySets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
            hGradWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, gradHistorySets[pass], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsGradientWeightInfo, nullptr};
            vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(hGradWrites.size()), hGradWrites.data(), 0, nullptr);
        }
    }

    return true;
}

bool HeatModelRuntime::createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes) {
    if (materialNodes.empty()) return false;

    return createStorageBuffer(
        memoryAllocator,
        vulkanDevice,
        materialNodes.data(),
        sizeof(heat::MaterialNode) * materialNodes.size(),
        materialBuffer,
        materialBufferOffset,
        nullptr
    ) == VK_SUCCESS;
}

void HeatModelRuntime::setSimResources(
    VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
    VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset, uint32_t gmlsCount) {
    this->simNodeBuffer = nodeBuffer;
    this->simNodeOffset = nodeOffset;
    this->simNodeCount = nodeCount;
    this->simGMLSInterfaceBuffer = gmlsBuffer;
    this->simGMLSInterfaceOffset = gmlsOffset;
    this->simGMLSInterfaceCount = gmlsCount;
}

void HeatModelRuntime::setVoronoiToSimNodeId(const std::vector<uint32_t>& mapping) {
    voronoiToSimNodeId = mapping;
}


bool HeatModelRuntime::ensureSimulationBuffers(uint32_t nodeCount) {
    if (this->simNodeCount == nodeCount && tempBufferA != VK_NULL_HANDLE) {
        return true;
    }

    cleanupSimulationBuffers();
    this->simNodeCount = nodeCount;

    if (nodeCount == 0) return true;

    // Allocate Temp A
    auto [handleA, offsetA] = memoryAllocator.allocate(
        nodeCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tempBufferA = handleA;
    tempBufferAOffset = offsetA;

    // Allocate Temp B
    auto [handleB, offsetB] = memoryAllocator.allocate(
        nodeCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tempBufferB = handleB;
    tempBufferBOffset = offsetB;

    // Allocate coupling accumulator 
    auto [handleFlux, offsetFlux] = memoryAllocator.allocate(
        nodeCount * sizeof(float) * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    contactAccumulatorBuffer = handleFlux;
    contactAccumulatorBufferOffset = offsetFlux;

    if (contactAccumulatorBuffer != VK_NULL_HANDLE) {
        const VkDeviceSize accumulatorSize = nodeCount * sizeof(float) * 2;
        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        if (cmd == VK_NULL_HANDLE) {
            return false;
        }
        vkCmdFillBuffer(cmd, contactAccumulatorBuffer, contactAccumulatorBufferOffset, accumulatorSize, 0u);
        if (!renderCommandPool.endCommands(cmd)) {
            std::cerr << "[HEAT-UPLOAD] contactAccumulatorFill failed"
                      << " simNodeCount=" << nodeCount
                      << " dstBuffer=" << contactAccumulatorBuffer
                      << " dstOffset=" << contactAccumulatorBufferOffset
                      << " size=" << accumulatorSize
                      << std::endl;
            return false;
        }
    }

    return tempBufferA != VK_NULL_HANDLE && tempBufferB != VK_NULL_HANDLE && contactAccumulatorBuffer != VK_NULL_HANDLE;
}

void HeatModelRuntime::cleanupSimulationBuffers() {
    freeBuffer(memoryAllocator, tempBufferA, tempBufferAOffset);
    freeBuffer(memoryAllocator, tempBufferB, tempBufferBOffset);
    freeBuffer(memoryAllocator, contactAccumulatorBuffer, contactAccumulatorBufferOffset);
}

void HeatModelRuntime::setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree) {
    stencilKDTree = std::move(kdTree);
}

void HeatModelRuntime::updateHistoryDescriptorOffset(uint32_t displayFrame, VkDeviceSize frameStride, uint32_t currentFrame) {
    const bool useB = (currentFrame % 2) == 1;
    const VkDescriptorSet historySet = useB ? surfaceHistoryComputeSetB : surfaceHistoryComputeSetA;
    const VkDescriptorSet gradientSet = useB ? surfaceGradientHistorySetB : surfaceGradientHistorySetA;

    if (historySet == VK_NULL_HANDLE ||
        gradientSet == VK_NULL_HANDLE ||
        historyBuffer == VK_NULL_HANDLE ||
        simNodeCount == 0) {
        return;
    }

    VkDeviceSize offset = historyBufferOffset + static_cast<VkDeviceSize>(displayFrame) * frameStride;
    VkDeviceSize size = static_cast<VkDeviceSize>(simNodeCount) * sizeof(float);

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = historyBuffer;
    bufferInfo.offset = offset;
    bufferInfo.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = historySet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), 1, &write, 0, nullptr);

    write.dstSet = gradientSet;
    vkUpdateDescriptorSets(vulkanDevice.getDevice(), 1, &write, 0, nullptr);
}
