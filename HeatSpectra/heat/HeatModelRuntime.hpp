#pragma once

#include "heat/HeatGpuStructs.hpp"
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
        VkBuffer gradientWeightBuffer,
        VkDeviceSize gradientWeightBufferOffset);

    void updateAllDescriptors(
        VkDescriptorSetLayout surfaceLayout,
        VkDescriptorSetLayout gradientLayout,
        VkDescriptorPool surfacePool,
        VkDescriptorSetLayout voronoiLayout,
        VkDescriptorPool voronoiPool,
        VkBuffer timeBuffer,
        VkDeviceSize timeBufferOffset,
        bool forceReallocate = false);

    VkDescriptorSet getSurfaceComputeSetA() const { return surfaceComputeSetA; }
    VkDescriptorSet getSurfaceComputeSetB() const { return surfaceComputeSetB; }
    VkDescriptorSet getSurfaceGradientComputeSetA() const { return surfaceGradientComputeSetA; }
    VkDescriptorSet getSurfaceGradientComputeSetB() const { return surfaceGradientComputeSetB; }

    void injectFixedTemperature(VkBuffer tempBuffer, VkDeviceSize tempBufferOffset, uint32_t nodeCount, float temperature) const;

    const SupportingHalfedge::IntrinsicMesh& getIntrinsicMesh() const { return intrinsicMesh; }

    bool createMaterialBuffer(const std::vector<heat::MaterialNode>& materialNodes);
    VkBuffer getMaterialBuffer() const { return materialBuffer; }
    VkDeviceSize getMaterialBufferOffset() const { return materialBufferOffset; }

    void setVoronoiResources(
        VkBuffer nodeBuffer, VkDeviceSize nodeOffset, uint32_t nodeCount,
        VkBuffer gmlsBuffer, VkDeviceSize gmlsOffset,
        VkBuffer seedBuffer, VkDeviceSize seedOffset);

    VkBuffer getVoronoiNodeBuffer() const { return voronoiNodeBuffer; }
    VkDeviceSize getVoronoiNodeOffset() const { return voronoiNodeOffset; }
    uint32_t getVoronoiNodeCount() const { return voronoiNodeCount; }
    VkBuffer getGMLSInterfaceBuffer() const { return gmlsInterfaceBuffer; }
    VkDeviceSize getGMLSInterfaceOffset() const { return gmlsInterfaceOffset; }
    VkBuffer getVoronoiSeedFlagsBuffer() const { return voronoiSeedFlagsBuffer; }
    VkDeviceSize getVoronoiSeedFlagsOffset() const { return voronoiSeedFlagsOffset; }

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

    float density = 1000.0f;
    float specificHeat = 1000.0f;
    float conductivity = 1.0f;

    uint32_t boundaryCondition = 0;
    float fixedTemperatureValue = 1.0f;
    VkBuffer surfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfaceBufferOffset = 0;

    VkBuffer gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceStencilBufferOffset = 0;
    VkBuffer gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceWeightBufferOffset = 0;
    VkBuffer gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset = 0;

    VkBuffer surfaceGradientBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfaceGradientBufferOffset = 0;

    VkDescriptorSet surfaceComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceComputeSetB = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientComputeSetB = VK_NULL_HANDLE;
    
    VkBuffer materialBuffer = VK_NULL_HANDLE;
    VkDeviceSize materialBufferOffset = 0;

    uint32_t voronoiNodeCount = 0;
    VkBuffer voronoiNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiNodeOffset = 0;
    VkBuffer gmlsInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsInterfaceOffset = 0;
    VkBuffer voronoiSeedFlagsBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiSeedFlagsOffset = 0;

    VkDescriptorSet voronoiDescriptorSetA = VK_NULL_HANDLE;
    VkDescriptorSet voronoiDescriptorSetB = VK_NULL_HANDLE;

    VkBuffer tempBufferA = VK_NULL_HANDLE;
    VkDeviceSize tempBufferAOffset = 0;
    VkBuffer tempBufferB = VK_NULL_HANDLE;
    VkDeviceSize tempBufferBOffset = 0;
    VkBuffer contactAccumulatorBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactAccumulatorBufferOffset = 0;
    uint32_t simNodeCount = 0;
    bool initialized = false;

    std::unique_ptr<StencilKDTree> stencilKDTree;
};
