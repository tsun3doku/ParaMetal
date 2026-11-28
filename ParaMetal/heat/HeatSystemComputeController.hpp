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

#include "serial/SerialTemperatureRuntime.hpp"
#include "heat/HeatProbeConfig.hpp"

class VulkanDevice;
class MemoryAllocator;
class CommandPool;
class HeatModelRuntime;
class TemperatureTelemetryReceiver;
class ConnectionTelemetryReceiver;
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
            HeatSimDefaults::fluidInitialTemperatureC,
            HeatSimDefaults::fluidInitialTemperatureC,
            HeatSimDefaults::fluidInitialTemperatureC,
            HeatSimDefaults::fluidInitialTemperatureC,
            HeatSimDefaults::fluidInitialTemperatureC,
            HeatSimDefaults::fluidInitialTemperatureC
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
        std::unordered_map<uint64_t, HeatProbeConfig> heatProbesBySourceKey;
        std::vector<uint32_t> fragmentInstanceIds;
        std::vector<uint32_t> fragmentSeedIds;
        std::vector<glm::dvec3> fragmentCentroids;
        std::vector<double> fragmentVolumes;

        std::vector<uint32_t> faceFragmentA;
        std::vector<uint32_t> faceFragmentB;
        std::vector<double> faceAreasA;
        std::vector<double> faceAreasB;
        std::vector<glm::dvec3> faceAreaVectorsA;
        std::vector<glm::dvec3> faceAreaVectorsB;
        std::vector<glm::dvec3> faceMomentsA;
        std::vector<glm::dvec3> faceMomentsB;
        std::vector<uint8_t> faceTypes;
        std::vector<uint32_t> faceDomainFaces;

        std::vector<uint32_t> boundaryFaceIndices;
        std::vector<uint32_t> faceLoopOffsets;
        std::vector<uint32_t> loopVertexOffsets;
        std::vector<glm::dvec3> loopVertices;

        std::vector<uint32_t> contactCouplingFaceA;
        std::vector<uint32_t> contactCouplingFragmentA;
        std::vector<uint32_t> contactCouplingFaceB;
        std::vector<uint32_t> contactCouplingFragmentB;
        std::vector<double> contactCouplingAreas;

        glm::vec3 domainMin{0.0f};
        glm::vec3 domainMax{0.0f};
        glm::vec3 volumeGridMin{0.0f};
        glm::ivec3 volumeGridDim{0, 0, 0};
        float volumeGridCellSize = 0.0f;

        VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
        VkDeviceSize seedIdGridBufferOffset = 0;
        std::vector<uint32_t> seedToSpaceIndex;
        std::vector<uint32_t> spaceToFragmentIndex;
        bool hasUniqueSeedToSpaceMapping = false;
        std::unordered_map<uint32_t, std::array<float, 16>> modelLocalToWorldByModelId;
        std::unordered_map<uint32_t, bool> modelPrescribedTemperatureEnabledByRuntimeId;
        std::unordered_map<uint32_t, float> modelPrescribedTemperaturesCByRuntimeId;
        uint64_t computeHash = 0;
        uint64_t geometryHash = 0;
    };

    HeatSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        CommandPool& transferCommandPool,
        uint32_t maxFramesInFlight);

    bool apply(uint64_t socketKey, const Config& config);
    bool remove(uint64_t socketKey);
    void syncSerialInputs();
    void updateSerialInputs();
    void publishProbeTelemetry(TemperatureTelemetryReceiver& receiver) const;
    void publishSerialTelemetry(ConnectionTelemetryReceiver& receiver) const;

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
    bool commitPending(uint64_t socketKey) noexcept;
    void cancelCandidate(uint64_t socketKey);
    void quiescedClear() noexcept;

private:
    bool retireCurrent(uint64_t socketKey);

    struct GenerationSlot {
        std::unique_ptr<HeatSystem> current;
        std::unique_ptr<HeatSystem> retiring;
        std::unique_ptr<HeatSystem> candidate;
    };

    bool configureHeatSystem(HeatSystem& system, const Config& config);
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

