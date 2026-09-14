#pragma once

#include "util/Units.hpp"

#include "heat/HeatGpuStructs.hpp"
#include "vulkan/VulkanExternalBuffer.hpp"
#include "cuda/CudaExternalBuffer.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>
#include <memory>

class HeatSystemPlayback;
class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class HeatModelRuntime {
public:
    HeatModelRuntime(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        const std::vector<glm::vec3>& surfacePositions,
        const std::vector<glm::vec3>& surfaceNormals,
        const std::vector<uint32_t>& surfaceTriangleIndices,
        CommandPool& renderCommandPool,
        float initialTemperatureC);
    ~HeatModelRuntime();

    void setMaterialProperties(float density, float specificHeat, float conductivity);
    void setWorldUnit(units::LengthUnit unit) { worldUnit = unit; }
    float getMetersPerWorldUnit() const { return units::physics::metersPerWorldUnit(worldUnit); }
    void setInitialTemperatureC(float temperatureC) { initialTemperatureC = temperatureC; }

    void cleanup();
    bool isInitialized() const { return initialized; }

    float getDensity() const { return density; }
    float getSpecificHeat() const { return specificHeat; }
    float getConductivity() const { return conductivity; }
    float getInitialTemperatureC() const { return initialTemperatureC; }
    void setPrescribedTemperature(bool enabled, float temperatureC) {
        prescribedTemperatureEnabled = enabled;
        prescribedTemperatureC = temperatureC;
    }
    bool hasPrescribedTemperature() const { return prescribedTemperatureEnabled; }
    float getPrescribedTemperatureC() const { return prescribedTemperatureC; }
    float getVolumetricPowerDensity() const { return volumetricPowerDensity; }

    size_t getSurfaceVertexCount() const { return surfacePositions.size(); }

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
        VkBuffer playbackBuffer,
        VkDeviceSize playbackBufferOffset,
        bool forceReallocate = false);

    VkDescriptorSet getSurfaceComputeSetA() const { return surfaceComputeSetA; }
    VkDescriptorSet getSurfaceComputeSetB() const { return surfaceComputeSetB; }
    VkDescriptorSet getSurfaceGradientComputeSetA() const { return surfaceGradientComputeSetA; }
    VkDescriptorSet getSurfaceGradientComputeSetB() const { return surfaceGradientComputeSetB; }
    VkDescriptorSet getSurfaceHistoryComputeSetA() const { return surfaceHistoryComputeSetA; }
    VkDescriptorSet getSurfaceHistoryComputeSetB() const { return surfaceHistoryComputeSetB; }
    VkDescriptorSet getSurfaceGradientHistorySetA() const { return surfaceGradientHistorySetA; }
    VkDescriptorSet getSurfaceGradientHistorySetB() const { return surfaceGradientHistorySetB; }

    const std::vector<glm::vec3>& getSurfacePositions() const { return surfacePositions; }
    const std::vector<glm::vec3>& getSurfaceNormals() const { return surfaceNormals; }
    const std::vector<uint32_t>& getSurfaceTriangleIndices() const { return surfaceTriangleIndices; }
    void setHistoryBuffer(VkBuffer buffer, VkDeviceSize offset, uint32_t frameCapacity);

    void initializePlayback(VulkanDevice& device, MemoryAllocator& allocator, uint32_t frameCapacity);
    HeatSystemPlayback* getPlayback() const { return playback.get(); }

    bool ensureSimulationBuffers(uint32_t nodeCount);
    void cleanupSimulationBuffers();
    VkBuffer getTempBufferA() const { return tempBufferA.getBuffer(); }
    VkBuffer getTempBufferB() const { return tempBufferB.getBuffer(); }
    const VulkanExternalBuffer& getExternalTempBufferA() const { return tempBufferA; }
    const VulkanExternalBuffer& getExternalTempBufferB() const { return tempBufferB; }
    CudaExternalBuffer& getCudaTempBufferA() { return cudaTempBufferA; }
    CudaExternalBuffer& getCudaTempBufferB() { return cudaTempBufferB; }
    uint32_t getSimNodeCount() const { return simNodeCount; }

    bool setVolumetricPowerDensity(float powerDensity);
    void updateHistoryDescriptorOffset(uint32_t displayFrame, VkDeviceSize frameStride, uint32_t currentFrame);

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    std::vector<glm::vec3> surfacePositions;
    std::vector<glm::vec3> surfaceNormals;
    std::vector<uint32_t> surfaceTriangleIndices;
    CommandPool& renderCommandPool;

    float density = HeatSimDefaults::density;
    float specificHeat = HeatSimDefaults::specificHeat;
    float conductivity = HeatSimDefaults::conductivity;
    units::LengthUnit worldUnit = units::defaultLengthUnit();
    float initialTemperatureC = HeatSimDefaults::ambientTemperatureC;
    float volumetricPowerDensity = 0.0f;
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
    VkDescriptorSet surfaceHistoryComputeSetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceHistoryComputeSetB = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientHistorySetA = VK_NULL_HANDLE;
    VkDescriptorSet surfaceGradientHistorySetB = VK_NULL_HANDLE;

    VulkanExternalBuffer tempBufferA;
    VulkanExternalBuffer tempBufferB;
    CudaExternalBuffer cudaTempBufferA;
    CudaExternalBuffer cudaTempBufferB;
    VkBuffer historyBuffer = VK_NULL_HANDLE;
    VkDeviceSize historyBufferOffset = 0;
    uint32_t historyBufferFrameCapacity = 0;
    uint32_t simNodeCount = 0;
    bool prescribedTemperatureEnabled = false;
    float prescribedTemperatureC = HeatSimDefaults::ambientTemperatureC;
    bool initialized = false;

    std::unique_ptr<HeatSystemPlayback> playback;
};
