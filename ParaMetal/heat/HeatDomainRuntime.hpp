#pragma once

#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

struct StencilKDTree;

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class HeatDomainRuntime {
public:
    virtual ~HeatDomainRuntime();
    virtual bool hasSurface() const = 0;

    // Simulation buffers
    bool ensureSimulationBuffers(uint32_t nodeCount);
    void cleanupSimulationBuffers();

    // Material
    bool createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes);
    VkBuffer getMaterialBuffer() const { return materialBuffer; }
    VkDeviceSize getMaterialBufferOffset() const { return materialBufferOffset; }

    // Sim resources (Voronoi node buffer + GMLS interface buffer)
    void setSimResources(
        VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
        VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset, uint32_t gmlsCount);
    VkBuffer getSimNodeBuffer() const { return simNodeBuffer; }
    VkDeviceSize getSimNodeOffset() const { return simNodeOffset; }
    VkBuffer getSimGMLSInterfaceBuffer() const { return simGMLSInterfaceBuffer; }
    VkDeviceSize getSimGMLSInterfaceOffset() const { return simGMLSInterfaceOffset; }
    uint32_t getSimGMLSInterfaceCount() const { return simGMLSInterfaceCount; }

    // Temperature ping-pong buffers
    VkBuffer getTempBufferA() const { return tempBufferA; }
    VkDeviceSize getTempBufferAOffset() const { return tempBufferAOffset; }
    VkBuffer getTempBufferB() const { return tempBufferB; }
    VkDeviceSize getTempBufferBOffset() const { return tempBufferBOffset; }

    // Contact accumulator
    VkBuffer getContactAccumulatorBuffer() const { return contactAccumulatorBuffer; }
    VkDeviceSize getContactAccumulatorBufferOffset() const { return contactAccumulatorBufferOffset; }

    // Voronoi-to-sim mapping
    void setVoronoiToSimNodeId(const std::vector<uint32_t>& mapping);
    uint32_t mapVoronoiNodeToSim(uint32_t voronoiNodeId) const {
        return voronoiNodeId < voronoiToSimNodeId.size() ? voronoiToSimNodeId[voronoiNodeId] : UINT32_MAX;
    }

    // Voronoi descriptor sets
    void updateVoronoiDescriptors(
        VkDescriptorSetLayout voronoiLayout,
        VkDescriptorPool voronoiPool,
        VkBuffer timeBuffer,
        VkDeviceSize timeBufferOffset,
        bool forceReallocate = false);
    VkDescriptorSet getVoronoiDescriptorSetA() const { return voronoiDescriptorSetA; }
    VkDescriptorSet getVoronoiDescriptorSetB() const { return voronoiDescriptorSetB; }

    // KDTree for spatial stencil lookups
    void setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree);
    StencilKDTree* getStencilKDTree() const { return stencilKDTree.get(); }

    uint32_t getSimNodeCount() const { return simNodeCount; }

protected:
    HeatDomainRuntime(VulkanDevice& device, MemoryAllocator& allocator, CommandPool& pool);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& renderCommandPool;

    VkBuffer tempBufferA = VK_NULL_HANDLE;
    VkDeviceSize tempBufferAOffset = 0;
    VkBuffer tempBufferB = VK_NULL_HANDLE;
    VkDeviceSize tempBufferBOffset = 0;
    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceSize materialBufferOffset = 0;
    VkBuffer simNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize simNodeOffset = 0;
    VkBuffer simGMLSInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize simGMLSInterfaceOffset = 0;
    uint32_t simGMLSInterfaceCount = 0;
    VkBuffer contactAccumulatorBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactAccumulatorBufferOffset = 0;
    VkDescriptorSet voronoiDescriptorSetA = VK_NULL_HANDLE;
    VkDescriptorSet voronoiDescriptorSetB = VK_NULL_HANDLE;
    std::vector<uint32_t> voronoiToSimNodeId;
    std::unique_ptr<StencilKDTree> stencilKDTree;
    uint32_t simNodeCount = 0;
};
