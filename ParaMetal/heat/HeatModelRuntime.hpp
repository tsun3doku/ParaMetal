#pragma once

#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include <memory>

struct StencilKDTree;

class HeatSystemPlayback;
class VulkanDevice;
class MemoryAllocator;
class CommandPool;
struct HeatProduct;

class HeatModelRuntime {
public:
    HeatModelRuntime(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
        CommandPool& renderCommandPool,
        float initialTemperature);
    ~HeatModelRuntime();

    void setBoundaryCondition(uint32_t bc);
    void setFixedTemperatureValue(float temperature);
    void setMaterialProperties(float density, float specificHeat, float conductivity);

    bool appendProduct(struct HeatProduct& product);
    void cleanup();
    bool isInitialized() const { return initialized; }

    uint32_t getBoundaryCondition() const { return boundaryCondition; }
    float getFixedTemperatureValue() const { return fixedTemperatureValue; }
    float getDensity() const { return density; }
    float getSpecificHeat() const { return specificHeat; }
    float getConductivity() const { return conductivity; }


    size_t getIntrinsicVertexCount() const { return intrinsicVertexCount; }



    void setGMLSSurfaceWeights(
        VkBuffer stencilBuffer,
        VkDeviceSize stencilBufferOffset,
        VkBuffer valueWeightBuffer,
        VkDeviceSize valueWeightBufferOffset,
        size_t valueWeightCount,
        VkBuffer gradientWeightBuffer,
        VkDeviceSize gradientWeightBufferOffset,
        size_t gradientWeightCount);

    bool updateAllDescriptors(
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
        bool forceReallocate = false);

    VkDescriptorSet getSurfaceComputeSetA() const { return surfaceComputeSetA; }
    VkDescriptorSet getSurfaceComputeSetB() const { return surfaceComputeSetB; }
    VkDescriptorSet getSurfaceGradientComputeSetA() const { return surfaceGradientComputeSetA; }
    VkDescriptorSet getSurfaceGradientComputeSetB() const { return surfaceGradientComputeSetB; }
    VkDescriptorSet getSurfaceHistoryComputeSetA() const { return surfaceHistoryComputeSetA; }
    VkDescriptorSet getSurfaceHistoryComputeSetB() const { return surfaceHistoryComputeSetB; }
    VkDescriptorSet getSurfaceGradientHistorySetA() const { return surfaceGradientHistorySetA; }
    VkDescriptorSet getSurfaceGradientHistorySetB() const { return surfaceGradientHistorySetB; }

    const SupportingHalfedge::IntrinsicMesh& getIntrinsicMesh() const { return intrinsicMesh; }

    bool createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes);
    VkBuffer getMaterialBuffer() const { return materialBuffer; }
    VkDeviceSize getMaterialBufferOffset() const { return materialBufferOffset; }

    void setSimResources(
        VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
        VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset, uint32_t gmlsCount);
    void setVoronoiToSimNodeId(const std::vector<uint32_t>& mapping);
    void setHistoryBuffer(VkBuffer buffer, VkDeviceSize offset, uint32_t frameCapacity);

    void initializePlayback(VulkanDevice& device, MemoryAllocator& allocator, uint32_t frameCapacity);
    HeatSystemPlayback* getPlayback() const { return playback.get(); }

    VkBuffer getSimNodeBuffer() const { return simNodeBuffer; }
    VkDeviceSize getSimNodeOffset() const { return simNodeOffset; }
    VkBuffer getSimGMLSInterfaceBuffer() const { return simGMLSInterfaceBuffer; }
    VkDeviceSize getSimGMLSInterfaceOffset() const { return simGMLSInterfaceOffset; }
    uint32_t mapVoronoiNodeToSim(uint32_t voronoiNodeId) const {
        return voronoiNodeId < voronoiToSimNodeId.size() ? voronoiToSimNodeId[voronoiNodeId] : UINT32_MAX;
    }

    VkDescriptorSet getVoronoiDescriptorSetA() const { return voronoiDescriptorSetA; }
    VkDescriptorSet getVoronoiDescriptorSetB() const { return voronoiDescriptorSetB; }

    bool ensureSimulationBuffers(uint32_t nodeCount);
    void cleanupSimulationBuffers();
    VkBuffer getTempBufferA() const { return tempBufferA; }
    VkDeviceSize getTempBufferAOffset() const { return tempBufferAOffset; }
    VkBuffer getTempBufferB() const { return tempBufferB; }
    VkDeviceSize getTempBufferBOffset() const { return tempBufferBOffset; }
    VkBuffer getContactAccumulatorBuffer() const { return contactAccumulatorBuffer; }
    VkDeviceSize getContactAccumulatorBufferOffset() const { return contactAccumulatorBufferOffset; }
    uint32_t getSimNodeCount() const { return simNodeCount; }

    void setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree);
    StencilKDTree* getStencilKDTree() const { return stencilKDTree.get(); }

    void updateHistoryDescriptorOffset(uint32_t displayFrame, VkDeviceSize frameStride, uint32_t currentFrame);

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    SupportingHalfedge::IntrinsicMesh intrinsicMesh{};
    CommandPool& renderCommandPool;
    size_t intrinsicVertexCount = 0;

    float density = HeatSimDefaults::density;
    float specificHeat = HeatSimDefaults::specificHeat;
    float conductivity = HeatSimDefaults::conductivity;

    uint32_t boundaryCondition = 0;
    float fixedTemperatureValue = HeatSimDefaults::ambientTemperature;


    VkBuffer gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceStencilBufferOffset = 0;
    VkBuffer gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceWeightBufferOffset = 0;
    size_t gmlsSurfaceWeightCount = 0;
    VkBuffer gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset = 0;
    size_t gmlsSurfaceGradientWeightCount = 0;



    VkDescriptorSet surfaceComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceComputeSetB = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetB = VK_NULL_HANDLE;
    // History sets are double-buffered (A/B) because updateHistoryDescriptorOffset
    // rewrites them every frame for the current display frame. A singleton set would
    // be mutated while the prior frame's command buffer (which bound it) is still
    // pending on the GPU - a descriptor-aliasing violation (VUID-vkUpdateDescriptorSets-
    // None-03047) and the source of device-lost on heavy churn. The A/B split matches
    // the existing pattern for surfaceComputeSetA/B and voronoiDescriptorSetA/B.
    VkDescriptorSet surfaceHistoryComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceHistoryComputeSetB = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientHistorySetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientHistorySetB = VK_NULL_HANDLE;

    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceSize materialBufferOffset = 0;

    VkBuffer simNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize simNodeOffset = 0;
    VkBuffer simGMLSInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize simGMLSInterfaceOffset = 0;
    std::vector<uint32_t> voronoiToSimNodeId;

    VkDescriptorSet voronoiDescriptorSetA = VK_NULL_HANDLE;
    VkDescriptorSet voronoiDescriptorSetB = VK_NULL_HANDLE;

    VkBuffer tempBufferA = VK_NULL_HANDLE;
    VkDeviceSize tempBufferAOffset = 0;
    VkBuffer tempBufferB = VK_NULL_HANDLE;
    VkDeviceSize tempBufferBOffset = 0;
    VkBuffer contactAccumulatorBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactAccumulatorBufferOffset = 0;
    VkBuffer historyBuffer = VK_NULL_HANDLE;
    VkDeviceSize historyBufferOffset = 0;
    uint32_t historyBufferFrameCapacity = 0;
    uint32_t simNodeCount = 0;
    uint32_t simGMLSInterfaceCount = 0;
    bool initialized = false;

    std::unique_ptr<StencilKDTree> stencilKDTree;
    std::unique_ptr<HeatSystemPlayback> playback;
};
