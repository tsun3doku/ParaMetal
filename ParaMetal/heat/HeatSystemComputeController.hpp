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
#include "runtime/TimelinePlaybackTarget.hpp"

class VulkanDevice;
class MemoryAllocator;
class CommandPool;
class HeatModelRuntime;
struct HeatProduct;

class HeatSystemComputeController : public TimelinePlaybackTarget {
public:
    struct Config {
        bool active = false;
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
        uint32_t solverMaxIterations = 0;         // 0 = keep heat solver default
        float solverConvergenceTolerance = 0.0f;  // 0 = keep heat solver default
        std::vector<std::vector<glm::vec3>> modelSurfacePositions;
        std::vector<std::vector<glm::vec3>> modelSurfaceNormals;
        std::vector<std::vector<uint32_t>> modelSurfaceTriangleIndices;
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
        topology::GlobalDomain domainVoronoiProduct;
        std::unordered_map<uint32_t, std::array<float, 16>> modelLocalToWorldByModelId;
        std::unordered_map<uint32_t, bool> modelPrescribedTemperatureEnabledByRuntimeId;
        std::unordered_map<uint32_t, float> modelPrescribedTemperaturesCByRuntimeId;
        uint64_t computeHash = 0;
        uint64_t structuralHash = 0;
    };

    HeatSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        CommandPool& transferCommandPool,
        uint32_t maxFramesInFlight);

    // Returns false on failure, previous generation stays active and caller skips publish
    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void disableAll();
    void syncSerialInputs();
    void updateSerialInputs();
    bool getSerialTemperatureStatus(uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const;
    bool isAnyHeatSystemActive() const;
    bool isAnyHeatSystemPaused() const;

    void setTimelinePlaying(bool playing) override;
    void resetTimeline() override;
    void scrubTimeline(uint32_t frame) override;

    std::vector<HeatSystem*> getActiveSystems() const;

    // Blocks until in-flight solves finish for current and retained systems
    void waitIdle();
    // Host-signals pending solve release waits for current and retained systems
    void releaseWaits();
    bool buildProduct(uint64_t socketKey, HeatProduct& outProduct);
    const HeatSystem* getSystem(uint64_t socketKey) const;
    HeatSystem* getSystem(uint64_t socketKey);
    const Config* getConfig(uint64_t socketKey) const;

private:
    void configureHeatSystem(HeatSystem& system, const Config& config);
    void applyRuntimeState(HeatSystem& system, const Config& config);
    std::unique_ptr<HeatSystem> buildHeatSystem();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& renderCommandPool;
    CommandPool& transferCommandPool;
    uint32_t maxFramesInFlight;

    bool timelinePlaying = false;
    uint32_t timelineResetCounter = 0;
    uint32_t timelineScrubFrame = std::numeric_limits<uint32_t>::max();

    std::unordered_map<uint64_t, std::unique_ptr<HeatSystem>> systemsBySocket;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
    std::vector<std::unique_ptr<HeatSystem>> retainedSystems;
    std::unordered_map<uint64_t, std::unique_ptr<SerialTemperatureRuntime>> serialRuntimes;
    std::unordered_map<uint64_t, uint64_t> lastSerialRevisions;
};
