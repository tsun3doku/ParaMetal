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

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

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

    bool createSurfaceBuffer();
    void cleanup();
    bool isInitialized() const { return initialized; }

    uint32_t getBoundaryCondition() const { return boundaryCondition; }
    float getFixedTemperatureValue() const { return fixedTemperatureValue; }
    float getDensity() const { return density; }
    float getSpecificHeat() const { return specificHeat; }
    float getConductivity() const { return conductivity; }


    size_t getIntrinsicVertexCount() const { return intrinsicVertexCount; }

    VkBuffer getSurfaceBuffer() const { return surfaceBuffer; }
    VkDeviceSize getSurfaceBufferOffset() const { return surfaceBufferOffset; }

    VkBuffer getSurfaceGradientBuffer() const { return surfaceGradientBuffer; }
    VkDeviceSize getSurfaceGradientBufferOffset() const { return surfaceGradientBufferOffset; }

    void setGMLSSurfaceWeights(
        VkBuffer stencilBuffer,
        VkDeviceSize stencilBufferOffset,
        VkBuffer valueWeightBuffer,
        VkDeviceSize valueWeightBufferOffset,
        size_t valueWeightCount,
        VkBuffer gradientWeightBuffer,
        VkDeviceSize gradientWeightBufferOffset,
        size_t gradientWeightCount);

    void updateAllDescriptors(
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

    const SupportingHalfedge::IntrinsicMesh& getIntrinsicMesh() const { return intrinsicMesh; }

    bool createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes);
    VkBuffer getMaterialBuffer() const { return materialBuffer; }
    VkDeviceSize getMaterialBufferOffset() const { return materialBufferOffset; }

    void setSimResources(
        VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
        VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset, uint32_t gmlsCount);
    void setVoronoiToSimNodeId(const std::vector<uint32_t>& mapping);
    void setHistoryBuffer(VkBuffer buffer, VkDeviceSize offset);

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
    VkBuffer surfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfaceBufferOffset = 0;

    VkBuffer gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceStencilBufferOffset = 0;
    VkBuffer gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceWeightBufferOffset = 0;
    size_t gmlsSurfaceWeightCount = 0;
    VkBuffer gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset = 0;
    size_t gmlsSurfaceGradientWeightCount = 0;

    VkBuffer surfaceGradientBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfaceGradientBufferOffset = 0;

    VkDescriptorSet surfaceComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceComputeSetB = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetB = VK_NULL_HANDLE;
    
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
    uint32_t simNodeCount = 0;
    uint32_t simGMLSInterfaceCount = 0;
    bool initialized = false;

    std::unique_ptr<StencilKDTree> stencilKDTree;
};
