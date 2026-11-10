#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <glm/glm.hpp>

#include "HeatSystem.hpp"
#include "HeatGpuStructs.hpp"
#include "HeatSystemPresets.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/GlobalDomainTopology.hpp"
#include "serial/SerialTemperatureRuntime.hpp"
#include "domain/FluidData.hpp"
#include "domain/HeatProbeData.hpp"

class VulkanDevice;
class MemoryAllocator;
class CommandPool;
class HeatModelRuntime;
struct HeatProduct;

class HeatSystemComputeController {
public:
    struct Config {
        float simulationDuration = 5.0f;
        bool hasFluid = false;
        float fluidDensity = 0.0f;
        float fluidSpecificHeat = 0.0f;
        float fluidConductivity = 0.0f;
        float fluidInitialTemperatureC = 0.0f;
    bool hasAdvection = false;
    VkBuffer advectionFaceFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize advectionFaceFluxBufferSize = 0;
    VkBuffer advectionBoundaryFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize advectionBoundaryFluxBufferSize = 0;
    uint32_t advectionFaceCount = 0;
    uint32_t advectionBoundaryFaceCount = 0;
        uint32_t domainBoundaryTypes[6] = {};
        float domainBoundaryTemperaturesC[6] = {
            HeatSimDefaults::ambientTemperatureC,
            HeatSimDefaults::ambientTemperatureC,
            HeatSimDefaults::ambientTemperatureC,
            HeatSimDefaults::ambientTemperatureC,
            HeatSimDefaults::ambientTemperatureC,
            HeatSimDefaults::ambientTemperatureC
        };
        float domainBoundaryHeatFluxes[6] = {};
        float domainBoundaryHeatTransferCoefficients[6] = {};
        float domainBoundaryInflowTemperaturesC[6] = {
            FluidData{}.initialTemperatureC,
            FluidData{}.initialTemperatureC,
            FluidData{}.initialTemperatureC,
            FluidData{}.initialTemperatureC,
            FluidData{}.initialTemperatureC,
            FluidData{}.initialTemperatureC
        };
        uint32_t solverMaxIterations = 0;
        float solverConvergenceTolerance = 0.0f;
        std::vector<std::vector<glm::vec3>> modelSurfacePositions;
        std::vector<std::vector<glm::vec3>> modelSurfaceNormals;
        std::vector<std::vector<uint32_t>> modelSurfaceTriangleIndices;
        std::vector<uint64_t> heatModelKeys;
        std::vector<uint32_t> modelRuntimeModelIds;
        std::unordered_map<uint32_t, float> modelInitialTemperaturesCByRuntimeId;
        std::unordered_map<uint32_t, float> modelVolumetricPowerDensitiesByRuntimeId;
        std::unordered_map<uint32_t, float> modelDensity;
        std::unordered_map<uint32_t, float> modelSpecificHeat;
        std::unordered_map<uint32_t, float> modelConductivity;
        units::LengthUnit worldUnit = units::defaultLengthUnit();
        std::unordered_map<uint64_t, bool> serialEnabledBySourceKey;
        std::unordered_map<uint64_t, std::string> serialPortNamesBySourceKey;
        std::unordered_map<uint64_t, uint32_t> serialBaudRatesBySourceKey;
        std::unordered_map<uint64_t, HeatProbeData> heatProbesBySourceKey;
        topology::GlobalDomain domainVoronoiProduct;
        std::unordered_map<uint32_t, std::array<float, 16>> modelLocalToWorldByModelId;
        std::unordered_map<uint32_t, bool> modelPrescribedTemperatureEnabledByRuntimeId;
        std::unordered_map<uint32_t, float> modelPrescribedTemperaturesCByRuntimeId;
        uint64_t computeHash = 0;
    };

    HeatSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        CommandPool& transferCommandPool,
        uint32_t maxFramesInFlight);

    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void syncSerialInputs();
    void updateSerialInputs();
    bool getSerialTemperatureStatus(uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const;
    bool getHeatProbeTemperatures(uint64_t sourceKey, std::vector<float>& outTemperaturesC) const;

    void waitIdle();
    void releaseWaits();
    bool buildProduct(uint64_t socketKey, HeatProduct& outProduct);
    const HeatSystem* getSystem(uint64_t socketKey) const;
    HeatSystem* getSystem(uint64_t socketKey);
    const Config* getConfig(uint64_t socketKey) const;
    uint64_t getCurrentComputeHash(uint64_t socketKey) const {
        const auto it = configuredConfigs.find(socketKey);
        return it != configuredConfigs.end() ? it->second.computeHash : 0;
    }

    bool hasAnyRetiring() const;
    void clearAllRetiring();
    void commitPending(uint64_t socketKey) noexcept;
    void cancelCandidate(uint64_t socketKey);
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct GenerationSlot {
        std::unique_ptr<HeatSystem> current;
        std::unique_ptr<HeatSystem> retiring;
        std::unique_ptr<HeatSystem> candidate;
    };

    void configureHeatSystem(HeatSystem& system, const Config& config);
    std::unique_ptr<HeatSystem> buildHeatSystem();
    const Config* getBuildConfig(uint64_t socketKey) const;

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& renderCommandPool;
    CommandPool& transferCommandPool;
    uint32_t maxFramesInFlight;

    std::unordered_map<uint64_t, GenerationSlot> generationSlots;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    std::unordered_map<uint64_t, Config> pendingConfigs;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
    std::unordered_map<uint64_t, std::unique_ptr<SerialTemperatureRuntime>> serialRuntimes;
    std::unordered_map<uint64_t, uint64_t> lastSerialRevisions;
};
