#include "HeatDomainRuntime.hpp"

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/VoronoiAdapters.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>

HeatDomainRuntime::HeatDomainRuntime(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    CommandPool& pool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      renderCommandPool(pool) {
}

HeatDomainRuntime::~HeatDomainRuntime() {
    cleanupSimulationBuffers();
    freeBuffer(memoryAllocator, materialBuffer, materialBufferOffset);
}

bool HeatDomainRuntime::ensureSimulationBuffers(uint32_t nodeCount) {
    if (this->simNodeCount == nodeCount && tempBufferA != VK_NULL_HANDLE) {
        return true;
    }

    cleanupSimulationBuffers();
    this->simNodeCount = nodeCount;

    if (nodeCount == 0) return true;

    auto [handleA, offsetA] = memoryAllocator.allocate(
        nodeCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tempBufferA = handleA;
    tempBufferAOffset = offsetA;

    auto [handleB, offsetB] = memoryAllocator.allocate(
        nodeCount * sizeof(float),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tempBufferB = handleB;
    tempBufferBOffset = offsetB;

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
            std::cerr << "[HEAT-UPLOAD] domainContactAccumulatorFill failed"
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

void HeatDomainRuntime::cleanupSimulationBuffers() {
    freeBuffer(memoryAllocator, tempBufferA, tempBufferAOffset);
    freeBuffer(memoryAllocator, tempBufferB, tempBufferBOffset);
    freeBuffer(memoryAllocator, contactAccumulatorBuffer, contactAccumulatorBufferOffset);
    tempBufferA = VK_NULL_HANDLE;
    tempBufferB = VK_NULL_HANDLE;
    contactAccumulatorBuffer = VK_NULL_HANDLE;
    simNodeCount = 0;
}

bool HeatDomainRuntime::createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes) {
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

void HeatDomainRuntime::setSimResources(
    VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
    VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset, uint32_t gmlsCount) {
    this->simNodeBuffer = nodeBuffer;
    this->simNodeOffset = nodeOffset;
    this->simNodeCount = nodeCount;
    this->simGMLSInterfaceBuffer = gmlsBuffer;
    this->simGMLSInterfaceOffset = gmlsOffset;
    this->simGMLSInterfaceCount = gmlsCount;
}

void HeatDomainRuntime::setVoronoiToSimNodeId(const std::vector<uint32_t>& mapping) {
    voronoiToSimNodeId = mapping;
}

void HeatDomainRuntime::updateVoronoiDescriptors(
    VkDescriptorSetLayout voronoiLayout,
    VkDescriptorPool voronoiPool,
    VkBuffer timeBuffer,
    VkDeviceSize timeBufferOffset,
    bool forceReallocate) {

    if (simNodeCount == 0 || simNodeBuffer == VK_NULL_HANDLE || simGMLSInterfaceBuffer == VK_NULL_HANDLE) {
        return;
    }

    if (forceReallocate) {
        voronoiDescriptorSetA = VK_NULL_HANDLE;
        voronoiDescriptorSetB = VK_NULL_HANDLE;
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

    if (voronoiDescriptorSetA == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo vNodeInfo{simNodeBuffer, simNodeOffset, simNodeCount * sizeof(voronoi::Node)};
    VkDescriptorBufferInfo vGmlsInfo{simGMLSInterfaceBuffer, simGMLSInterfaceOffset, simGMLSInterfaceCount * sizeof(voronoi::GMLSInterface)};
    VkDescriptorBufferInfo vMatInfo{materialBuffer, materialBufferOffset, simNodeCount * sizeof(heat::MaterialNode)};
    VkDescriptorBufferInfo timeInfo{timeBuffer, timeBufferOffset, sizeof(heat::SimPlaybackUniform)};

    const VkDescriptorSet vSets[2] = { voronoiDescriptorSetA, voronoiDescriptorSetB };
    const VkBuffer tempBuffers[2] = { tempBufferA, tempBufferB };
    const VkDeviceSize tempOffsets[2] = { tempBufferAOffset, tempBufferBOffset };

    for (uint32_t pass = 0; pass < 2; ++pass) {
        VkDescriptorBufferInfo nodeTempInfo{ tempBuffers[pass], tempOffsets[pass], simNodeCount * sizeof(float) };
        VkDescriptorBufferInfo nodeNextTempInfo{ tempBuffers[1 - pass], tempOffsets[1 - pass], simNodeCount * sizeof(float) };
        VkDescriptorBufferInfo fluxInfo{ contactAccumulatorBuffer, contactAccumulatorBufferOffset, simNodeCount * sizeof(float) * 2 };

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

void HeatDomainRuntime::setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree) {
    stencilKDTree = std::move(kdTree);
}
