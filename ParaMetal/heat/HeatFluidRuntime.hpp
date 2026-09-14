#pragma once

#include "vulkan/VulkanExternalBuffer.hpp"
#include "cuda/CudaExternalBuffer.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "heat/HeatSystemPlayback.hpp"
#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;

class HeatFluidRuntime {
public:
    HeatFluidRuntime();
    ~HeatFluidRuntime();

    HeatFluidRuntime(const HeatFluidRuntime&) = delete;
    HeatFluidRuntime& operator=(const HeatFluidRuntime&) = delete;

    void setMaterialProperties(float density, float specificHeat, float conductivity, float initialTemperatureC);

    float getDensity() const { return density; }
    float getSpecificHeat() const { return specificHeat; }
    float getConductivity() const { return conductivity; }
    float getInitialTemperatureC() const { return initialTemperatureC; }
    void setInitialTemperatureC(float temperatureC) { initialTemperatureC = temperatureC; }
    float getInletTemperatureC() const { return inletTemperatureC; }
    void setInletTemperatureC(float temperatureC) { inletTemperatureC = temperatureC; }
    const glm::vec3& getPrescribedVelocity() const { return prescribedVelocity; }
    void setPrescribedVelocity(const glm::vec3& v) { prescribedVelocity = v; }

    uint32_t getNodeCount() const { return nodeCount; }

    bool ensureSimulationBuffers(uint32_t count, VulkanDevice& device);
    void cleanupSimulationBuffers();
    void cleanup();

    void initializePlayback(VulkanDevice& device, MemoryAllocator& allocator, uint32_t frameCapacity);
    HeatSystemPlayback* getPlayback() const { return playback.get(); }

    VkBuffer getTempBufferA() const { return tempBufferA.getBuffer(); }
    VkBuffer getTempBufferB() const { return tempBufferB.getBuffer(); }
    const VulkanExternalBuffer& getExternalTempBufferA() const { return tempBufferA; }
    const VulkanExternalBuffer& getExternalTempBufferB() const { return tempBufferB; }
    CudaExternalBuffer& getCudaTempBufferA() { return cudaTempBufferA; }
    CudaExternalBuffer& getCudaTempBufferB() { return cudaTempBufferB; }

private:
    float density = HeatSimDefaults::fluidDensity;
    float specificHeat = HeatSimDefaults::fluidSpecificHeat;
    float conductivity = HeatSimDefaults::fluidConductivity;
    float initialTemperatureC = HeatSimDefaults::ambientTemperatureC;
    float inletTemperatureC = HeatSimDefaults::ambientTemperatureC;
    glm::vec3 prescribedVelocity{0.0f};

    uint32_t nodeCount = 0;

    VulkanExternalBuffer tempBufferA;
    VulkanExternalBuffer tempBufferB;
    CudaExternalBuffer cudaTempBufferA;
    CudaExternalBuffer cudaTempBufferB;

    std::unique_ptr<HeatSystemPlayback> playback;
};
