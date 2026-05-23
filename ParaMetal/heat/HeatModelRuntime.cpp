#include "HeatModelRuntime.hpp"

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
    initialized = createSurfaceBuffer();
}

HeatModelRuntime::~HeatModelRuntime() {
    cleanup();
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

bool HeatModelRuntime::createSurfaceBuffer() {
    cleanup();
    initialized = false;

    intrinsicVertexCount = 0;

    const float initialTemperatureValue = fixedTemperatureValue;
    constexpr float normalEpsilon = 1e-12f;

    std::vector<heat::SurfacePoint> surfacePoints;
    std::vector<uint32_t> indices;
    if (intrinsicMesh.vertices.empty() || intrinsicMesh.indices.size() < 3) {
        return false;
    }

    intrinsicVertexCount = intrinsicMesh.vertices.size();
    indices = intrinsicMesh.indices;

    std::vector<glm::vec3> positions(intrinsicVertexCount);
    for (size_t i = 0; i < intrinsicVertexCount; ++i) {
        positions[i] = intrinsicMesh.vertices[i].position;
    }
    const std::vector<float> vertexAreas = computeVertexAreas(positions, indices);

    surfacePoints.resize(intrinsicVertexCount);
    for (size_t i = 0; i < intrinsicVertexCount; ++i) {
        surfacePoints[i].position = positions[i];
        surfacePoints[i].temperature = initialTemperatureValue;
        surfacePoints[i].normal = intrinsicMesh.vertices[i].normal;
        surfacePoints[i].area = (i < vertexAreas.size()) ? vertexAreas[i] : 0.0f;
        surfacePoints[i].color = glm::vec4(1.0f);
    }

    if (surfacePoints.empty()) {
        return false;
    }

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (uploadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            surfacePoints.data(),
            sizeof(heat::SurfacePoint) * surfacePoints.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
            storageAlignment,
            surfaceBuffer,
            surfaceBufferOffset) != VK_SUCCESS) {
        return false;
    }

    const VkDeviceSize gradientBufferSize = sizeof(glm::vec4) * surfacePoints.size();
    std::vector<glm::vec4> zeroGradients(surfacePoints.size(), glm::vec4(0.0f));

    if (uploadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            zeroGradients.data(),
            gradientBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            surfaceGradientBuffer,
            surfaceGradientBufferOffset) != VK_SUCCESS) {
        return false;
    }

    initialized = true;
    return true;
}

void HeatModelRuntime::cleanup() {
    freeBuffer(memoryAllocator, surfaceBuffer, surfaceBufferOffset);
    freeBuffer(memoryAllocator, materialBuffer, materialBufferOffset);
    freeBuffer(memoryAllocator, surfaceGradientBuffer, surfaceGradientBufferOffset);

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

    cleanupSimulationBuffers();
    initialized = false;
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

void HeatModelRuntime::updateAllDescriptors(
    VkDescriptorSetLayout surfaceLayout,
    VkDescriptorSetLayout gradientLayout,
    VkDescriptorPool surfacePool,
    VkDescriptorSetLayout voronoiLayout,
    VkDescriptorPool voronoiPool,
    VkBuffer timeBuffer,
    VkDeviceSize timeBufferOffset,
    bool forceReallocate) {
    const size_t intrinsicVertexCount = getIntrinsicVertexCount();
    if (intrinsicVertexCount == 0 ||
        gmlsSurfaceStencilBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceWeightBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceGradientWeightBuffer == VK_NULL_HANDLE ||
        surfaceBuffer == VK_NULL_HANDLE ||
        surfaceGradientBuffer == VK_NULL_HANDLE) {
        return;
    }

    if (forceReallocate) {
        surfaceComputeSetA = VK_NULL_HANDLE;
        surfaceComputeSetB = VK_NULL_HANDLE;
        surfaceGradientComputeSetA = VK_NULL_HANDLE;
        surfaceGradientComputeSetB = VK_NULL_HANDLE;
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

    if (voronoiDescriptorSetA == VK_NULL_HANDLE || voronoiDescriptorSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> vLayouts = { voronoiLayout, voronoiLayout };
        VkDescriptorSetAllocateInfo vAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, voronoiPool, 2, vLayouts.data()};
        VkDescriptorSet vSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &vAllocInfo, vSets) == VK_SUCCESS) {
            voronoiDescriptorSetA = vSets[0];
            voronoiDescriptorSetB = vSets[1];
        }
    }

    if (surfaceComputeSetA == VK_NULL_HANDLE || voronoiDescriptorSetA == VK_NULL_HANDLE) return;

    // Common info
    VkDescriptorBufferInfo surfaceInfo{ surfaceBuffer, surfaceBufferOffset, sizeof(heat::SurfacePoint) * intrinsicVertexCount };
    VkDescriptorBufferInfo gradientInfo{ surfaceGradientBuffer, surfaceGradientBufferOffset, sizeof(glm::vec4) * intrinsicVertexCount };
    VkDescriptorBufferInfo timeInfo{ timeBuffer, timeBufferOffset, sizeof(heat::TimeUniform) };
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

        // Voronoi Diffusion
        std::array<VkWriteDescriptorSet, 7> vWrites{};
        vWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vNodeInfo, nullptr};
        vWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vGmlsInfo, nullptr};
        vWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vMatInfo, nullptr};
        vWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 3, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &timeInfo, nullptr};
        vWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        vWrites[5] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeNextTempInfo, nullptr};
        vWrites[6] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSets[pass], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fluxInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(vWrites.size()), vWrites.data(), 0, nullptr);
    }
}

bool HeatModelRuntime::createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes) {
    if (materialNodes.empty()) return false;

    freeBuffer(memoryAllocator, materialBuffer, materialBufferOffset);

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    return uploadDeviceBuffer(
        memoryAllocator,
        renderCommandPool,
        materialNodes.data(),
        sizeof(heat::MaterialNode) * materialNodes.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        storageAlignment,
        materialBuffer,
        materialBufferOffset
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
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tempBufferA = handleA;
    tempBufferAOffset = offsetA;

    // Allocate Temp B 
    auto [handleB, offsetB] = memoryAllocator.allocate(
        nodeCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
