#include "HeatModelRuntime.hpp"

#include "heat/HeatSystemPlayback.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

HeatModelRuntime::HeatModelRuntime(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    const std::vector<glm::vec3>& surfacePositions,
    const std::vector<glm::vec3>& surfaceNormals,
    const std::vector<uint32_t>& surfaceTriangleIndices,
    float initialTemperatureValueC)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      surfacePositions(surfacePositions),
      surfaceNormals(surfaceNormals),
      surfaceTriangleIndices(surfaceTriangleIndices),
      initialTemperatureC(initialTemperatureValueC) {
    initialized = !surfacePositions.empty() &&
        surfaceNormals.size() == surfacePositions.size() &&
        !surfaceTriangleIndices.empty();
}

HeatModelRuntime::~HeatModelRuntime() {
    cleanup();
}

void HeatModelRuntime::setMaterialProperties(float density, float specificHeat, float conductivity) {
    this->density = density;
    this->specificHeat = specificHeat;
    this->conductivity = conductivity;
}

void HeatModelRuntime::retainSurfaceBuffers(
    VkBuffer surface,
    VkDeviceSize surfaceOffset,
    VkBuffer gradient,
    VkDeviceSize gradientOffset) {
    if (surfaceBuffer == surface && surfaceGradientBuffer == gradient) {
        return;
    }
    releaseSurfaceBuffers();
    surfaceBuffer = surface;
    surfaceBufferOffset = surfaceOffset;
    surfaceGradientBuffer = gradient;
    surfaceGradientBufferOffset = gradientOffset;
}

void HeatModelRuntime::releaseSurfaceBuffers() {
    freeBuffer(memoryAllocator, surfaceBuffer, surfaceBufferOffset);
    freeBuffer(memoryAllocator, surfaceGradientBuffer, surfaceGradientBufferOffset);
}

void HeatModelRuntime::cleanup() {
    releaseSurfaceBuffers();
    if (playback) {
        playback->cleanup();
        playback.reset();
    }

    surfaceComputeSetA = VK_NULL_HANDLE;
    surfaceComputeSetB = VK_NULL_HANDLE;
    surfaceGradientComputeSetA = VK_NULL_HANDLE;
    surfaceGradientComputeSetB = VK_NULL_HANDLE;
    surfaceHistoryComputeSetA = VK_NULL_HANDLE;
    surfaceHistoryComputeSetB = VK_NULL_HANDLE;
    surfaceGradientHistorySetA = VK_NULL_HANDLE;
    surfaceGradientHistorySetB = VK_NULL_HANDLE;

    historyBuffer = VK_NULL_HANDLE;
    historyBufferOffset = 0;
    historyBufferFrameCapacity = 0;

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
    VkBuffer playbackBuffer,
    VkDeviceSize playbackBufferOffset,
    bool forceReallocate) {
    const size_t surfaceVertexCount = getSurfaceVertexCount();
    if (surfaceVertexCount == 0 ||
        gmlsSurfaceStencilBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceWeightBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceGradientWeightBuffer == VK_NULL_HANDLE ||
        surfaceBuffer == VK_NULL_HANDLE ||
        surfaceGradientBuffer == VK_NULL_HANDLE ||
        playbackBuffer == VK_NULL_HANDLE ||
        historyBuffer == VK_NULL_HANDLE ||
        !tempBufferA.isValid() ||
        !tempBufferB.isValid() ||
        simNodeCount == 0 ||
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

    if (surfaceComputeSetA == VK_NULL_HANDLE ||
        surfaceGradientComputeSetA == VK_NULL_HANDLE ||
        surfaceHistoryComputeSetA == VK_NULL_HANDLE ||
        surfaceGradientHistorySetA == VK_NULL_HANDLE) return false;

    // Common info
    VkDescriptorBufferInfo surfaceInfo{ surfaceBuffer, surfaceBufferOffset, sizeof(heat::SurfacePoint) * surfaceVertexCount };
    VkDescriptorBufferInfo gradientInfo{ surfaceGradientBuffer, surfaceGradientBufferOffset, sizeof(glm::vec4) * surfaceVertexCount };
    VkDescriptorBufferInfo playbackInfo{ playbackBuffer, playbackBufferOffset, sizeof(heat::SimPlaybackUniform) };
    VkDescriptorBufferInfo gmlsStencilInfo{ gmlsSurfaceStencilBuffer, gmlsSurfaceStencilBufferOffset, surfaceVertexCount * sizeof(heat::GMLSSurfaceStencil) };
    VkDescriptorBufferInfo gmlsValueWeightInfo{ gmlsSurfaceWeightBuffer, gmlsSurfaceWeightBufferOffset, gmlsSurfaceWeightCount * sizeof(heat::GMLSSurfaceWeight) };
    VkDescriptorBufferInfo gmlsGradientWeightInfo{ gmlsSurfaceGradientWeightBuffer, gmlsSurfaceGradientWeightBufferOffset, gmlsSurfaceGradientWeightCount * sizeof(heat::GMLSSurfaceGradientWeight) };
    
    const VkDescriptorSet sTempSets[2] = { surfaceComputeSetA, surfaceComputeSetB };
    const VkDescriptorSet sGradSets[2] = { surfaceGradientComputeSetA, surfaceGradientComputeSetB };
    const VkBuffer tempBuffers[2] = { tempBufferA.getBuffer(), tempBufferB.getBuffer() };
    const VkDeviceSize tempOffsets[2] = { 0, 0 };

    for (uint32_t pass = 0; pass < 2; ++pass) {
        VkDescriptorBufferInfo nodeTempInfo{ tempBuffers[pass], tempOffsets[pass], simNodeCount * sizeof(float) };

        // Surface Temperature Updates
        std::array<VkWriteDescriptorSet, 4> sTempWrites{};
        sTempWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        sTempWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
        sTempWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
        sTempWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sTempSets[pass], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsValueWeightInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(sTempWrites.size()), sTempWrites.data(), 0, nullptr);

        // Surface Gradient Updates
        std::array<VkWriteDescriptorSet, 5> sGradWrites{};
        sGradWrites[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nodeTempInfo, nullptr};
        sGradWrites[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &surfaceInfo, nullptr};
        sGradWrites[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gradientInfo, nullptr};
        sGradWrites[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsStencilInfo, nullptr};
        sGradWrites[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, sGradSets[pass], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gmlsGradientWeightInfo, nullptr};
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), 5, sGradWrites.data(), 0, nullptr);
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

bool HeatModelRuntime::setVolumetricPowerDensity(float powerDensity) {
    if (!std::isfinite(powerDensity)) return false;
    volumetricPowerDensity = powerDensity;
    return true;
}


bool HeatModelRuntime::ensureSimulationBuffers(uint32_t nodeCount) {
    if (this->simNodeCount == nodeCount && tempBufferA.isValid() && tempBufferB.isValid()) {
        return true;
    }

    cleanupSimulationBuffers();
    this->simNodeCount = nodeCount;

    if (nodeCount == 0) return true;

    const VkDeviceSize byteSize = nodeCount * sizeof(float);
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!tempBufferA.initialize(vulkanDevice, byteSize, usage)) {
        std::cerr << "[HeatSystem-Diag] ensureSimulationBuffers FAILED: tempBufferA.initialize"
                  << " nodeCount=" << nodeCount << " byteSize=" << byteSize << std::endl;
        return false;
    }
    if (!tempBufferB.initialize(vulkanDevice, byteSize, usage)) {
        std::cerr << "[HeatSystem-Diag] ensureSimulationBuffers FAILED: tempBufferB.initialize"
                  << " nodeCount=" << nodeCount << " byteSize=" << byteSize << std::endl;
        return false;
    }
    return true;
}

void HeatModelRuntime::cleanupSimulationBuffers() {
    cudaTempBufferA.cleanup();
    cudaTempBufferB.cleanup();
    tempBufferA.cleanup();
    tempBufferB.cleanup();
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
