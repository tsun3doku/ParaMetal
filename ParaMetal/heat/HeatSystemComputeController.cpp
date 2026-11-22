#include "HeatSystemComputeController.hpp"
#include "telemetry/RuntimeTelemetry.hpp"

#include <iostream>
#include "HeatSystem.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProduct.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/GeometryUtils.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <cassert>
#include <unordered_set>

static HeatSystemComputeController::Config makeRetainedConfig(
    const HeatSystemComputeController::Config& config) {
    HeatSystemComputeController::Config retained{};
    retained.simulationDuration = config.simulationDuration;
    retained.hasFluid = config.hasFluid;
    retained.fluidDensity = config.fluidDensity;
    retained.fluidSpecificHeat = config.fluidSpecificHeat;
    retained.fluidConductivity = config.fluidConductivity;
    retained.fluidInitialTemperatureC = config.fluidInitialTemperatureC;
    retained.hasAdvection = config.hasAdvection;
    retained.advectionFaceFluxBuffer = config.advectionFaceFluxBuffer;
    retained.advectionFaceFluxBufferSize = config.advectionFaceFluxBufferSize;
    retained.advectionBoundaryFluxBuffer = config.advectionBoundaryFluxBuffer;
    retained.advectionBoundaryFluxBufferSize = config.advectionBoundaryFluxBufferSize;
    retained.advectionFaceCount = config.advectionFaceCount;
    retained.advectionBoundaryFaceCount = config.advectionBoundaryFaceCount;
    for (size_t i = 0; i < 6; ++i) {
        retained.domainBoundaryTypes[i] = config.domainBoundaryTypes[i];
        retained.domainBoundaryTemperaturesC[i] = config.domainBoundaryTemperaturesC[i];
        retained.domainBoundaryHeatFluxes[i] = config.domainBoundaryHeatFluxes[i];
        retained.domainBoundaryHeatTransferCoefficients[i] = config.domainBoundaryHeatTransferCoefficients[i];
        retained.domainBoundaryInflowTemperaturesC[i] = config.domainBoundaryInflowTemperaturesC[i];
    }
    retained.solverMaxIterations = config.solverMaxIterations;
    retained.solverConvergenceTolerance = config.solverConvergenceTolerance;
    retained.modelSurfacePositions = config.modelSurfacePositions;
    retained.modelSurfaceNormals = config.modelSurfaceNormals;
    retained.modelSurfaceTriangleIndices = config.modelSurfaceTriangleIndices;
    retained.heatModelKeys = config.heatModelKeys;
    retained.modelRuntimeModelIds = config.modelRuntimeModelIds;
    retained.modelInitialTemperaturesCByRuntimeId = config.modelInitialTemperaturesCByRuntimeId;
    retained.modelVolumetricPowerDensitiesByRuntimeId = config.modelVolumetricPowerDensitiesByRuntimeId;
    retained.modelDensity = config.modelDensity;
    retained.modelSpecificHeat = config.modelSpecificHeat;
    retained.modelConductivity = config.modelConductivity;
    retained.worldUnit = config.worldUnit;
    retained.serialEnabledBySourceKey = config.serialEnabledBySourceKey;
    retained.serialPortNamesBySourceKey = config.serialPortNamesBySourceKey;
    retained.serialBaudRatesBySourceKey = config.serialBaudRatesBySourceKey;
    retained.heatProbesBySourceKey = config.heatProbesBySourceKey;
    retained.modelLocalToWorldByModelId = config.modelLocalToWorldByModelId;
    retained.modelPrescribedTemperatureEnabledByRuntimeId = config.modelPrescribedTemperatureEnabledByRuntimeId;
    retained.modelPrescribedTemperaturesCByRuntimeId = config.modelPrescribedTemperaturesCByRuntimeId;
    retained.computeHash = config.computeHash;
    retained.geometryHash = config.geometryHash;
    return retained;
}

HeatSystemComputeController::HeatSystemComputeController(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool,
    CommandPool& transferCommandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      renderCommandPool(renderCommandPool),
      transferCommandPool(transferCommandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

bool HeatSystemComputeController::configureHeatSystem(HeatSystem& system, const Config& config) {
    system.clearGlobalVoronoiInput();
    if (config.hasFluid) {
        system.setHeatFluid(
            config.fluidDensity,
            config.fluidSpecificHeat,
            config.fluidConductivity,
            config.fluidInitialTemperatureC);
    } else {
        system.clearHeatFluid();
    }
    if (config.hasAdvection) {
        system.setAdvectionInput(
            config.advectionFaceFluxBuffer,
            config.advectionFaceFluxBufferSize,
            config.advectionBoundaryFluxBuffer,
            config.advectionBoundaryFluxBufferSize,
            config.advectionFaceCount,
            config.advectionBoundaryFaceCount);
    } else {
        system.clearAdvectionInput();
    }
    if (!config.fragmentInstanceIds.empty()) {
        if (!system.setGlobalVoronoiInput(
                config.fragmentInstanceIds,
                config.fragmentSeedIds,
                config.fragmentCentroids,
                config.fragmentVolumes,
                config.faceFragmentA,
                config.faceFragmentB,
                config.faceAreasA,
                config.faceAreasB,
                config.faceAreaVectorsA,
                config.faceAreaVectorsB,
                config.faceMomentsA,
                config.faceMomentsB,
                config.faceTypes,
                config.faceDomainFaces,
                config.boundaryFaceIndices,
                config.contactCouplingFaceA,
                config.contactCouplingFragmentA,
                config.contactCouplingFaceB,
                config.contactCouplingFragmentB,
                config.contactCouplingAreas,
                config.modelLocalToWorldByModelId,
                config.seedIdGridBuffer,
                config.seedIdGridBufferOffset,
                config.seedToSpaceIndex,
                config.spaceToFragmentIndex,
                config.hasUniqueSeedToSpaceMapping,
                config.volumeGridDim)) {
            return false;
        }
    }
    system.setHeatModels(
        config.modelSurfacePositions,
        config.modelSurfaceNormals,
        config.modelSurfaceTriangleIndices,
        config.modelRuntimeModelIds,
        config.modelInitialTemperaturesCByRuntimeId,
        config.modelVolumetricPowerDensitiesByRuntimeId,
        config.modelPrescribedTemperatureEnabledByRuntimeId,
        config.modelPrescribedTemperaturesCByRuntimeId,
        config.modelDensity,
        config.modelSpecificHeat,
        config.modelConductivity,
        config.worldUnit);
    system.setParams(config.simulationDuration);
    system.setDomainBoundaryConditions(
        config.domainBoundaryTypes,
        config.domainBoundaryTemperaturesC,
        config.domainBoundaryHeatFluxes,
        config.domainBoundaryHeatTransferCoefficients,
        config.domainBoundaryInflowTemperaturesC);
    system.setGlobalSolverParameters(config.solverMaxIterations, config.solverConvergenceTolerance);
    return true;
}

bool HeatSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
            configIt->second.heatProbesBySourceKey = config.heatProbesBySourceKey;
        configIt->second.serialEnabledBySourceKey = config.serialEnabledBySourceKey;
        configIt->second.serialPortNamesBySourceKey = config.serialPortNamesBySourceKey;
        configIt->second.serialBaudRatesBySourceKey = config.serialBaudRatesBySourceKey;
        if (HeatSystem* system = getSystem(socketKey)) {
            system->syncHeatProbes(config.heatProbesBySourceKey);
        }
        syncSerialInputs();
        cancelCandidate(socketKey);
        return true;
    }

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    const auto slotIt = generationSlots.find(socketKey);
    if (slotIt != generationSlots.end() &&
        (slotIt->second.retiring || slotIt->second.candidate)) {
        return false;
    }

    std::unique_ptr<HeatSystem> candidate = buildHeatSystem();
    if (!candidate) {
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    if (!configureHeatSystem(*candidate, config) || !candidate->ensureConfigured()) {
        generationSlots[socketKey].retiring = std::move(candidate);
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }
    candidate->syncHeatProbes(config.heatProbesBySourceKey);

    GenerationSlot& slot = generationSlots[socketKey];
    configuredConfigs.try_emplace(socketKey);
    pendingConfigs[socketKey] = makeRetainedConfig(config);
    slot.candidate = std::move(candidate);

    syncSerialInputs();
    for (const auto& [sourceKey, portName] : config.serialPortNamesBySourceKey) {
        auto& runtime = serialRuntimes[sourceKey];
        if (!runtime) {
            runtime = std::make_unique<SerialTemperatureRuntime>();
        }
        runtime->configure(
            config.serialEnabledBySourceKey.at(sourceKey),
            portName,
            config.serialBaudRatesBySourceKey.at(sourceKey));
    }
    return true;
}

void HeatSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    failedComputeHashes.erase(socketKey);
    retireCurrent(socketKey);
    syncSerialInputs();
}

bool HeatSystemComputeController::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : generationSlots) {
        (void)socketKey;
        if (slot.retiring) return true;
    }
    return false;
}

void HeatSystemComputeController::clearAllRetiring() {
    for (auto it = generationSlots.begin(); it != generationSlots.end();) {
        GenerationSlot& slot = it->second;
        slot.retiring.reset();
        if (configuredConfigs.find(it->first) == configuredConfigs.end()) {
            slot.current.reset();
            slot.candidate.reset();
        }
        if (!slot.current && !slot.candidate && !slot.retiring) {
            it = generationSlots.erase(it);
        } else {
            ++it;
        }
    }
}

void HeatSystemComputeController::commitPending(uint64_t socketKey) noexcept {
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return;
    }
    GenerationSlot& slot = it->second;
    assert(slot.retiring == nullptr);
    if (slot.retiring != nullptr) {
        return;
    }
    if (slot.candidate) {
        slot.retiring = std::move(slot.current);
        slot.current = std::move(slot.candidate);
    }

    const auto pending = pendingConfigs.find(socketKey);
    if (pending != pendingConfigs.end()) {
        const auto committed = configuredConfigs.find(socketKey);
        if (committed != configuredConfigs.end()) {
            committed->second = std::move(pending->second);
        }
        pendingConfigs.erase(pending);
    }
    failedComputeHashes.erase(socketKey);
}

void HeatSystemComputeController::cancelCandidate(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it != generationSlots.end() && it->second.candidate) {
        assert(it->second.retiring == nullptr);
        if (!it->second.retiring) {
            it->second.retiring = std::move(it->second.candidate);
        }
    }
    const bool hasCurrent = it != generationSlots.end() && it->second.current;
    const auto pending = pendingConfigs.find(socketKey);
    if (pending != pendingConfigs.end()) {
        pendingConfigs.erase(pending);
        const auto committed = configuredConfigs.find(socketKey);
        if (committed != configuredConfigs.end() &&
            committed->second.computeHash == 0 &&
            !hasCurrent) {
            configuredConfigs.erase(committed);
        }
    }
}

void HeatSystemComputeController::retireCurrent(uint64_t socketKey) {
    configuredConfigs.erase(socketKey);
    pendingConfigs.erase(socketKey);

    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return;
    }
    GenerationSlot& slot = it->second;
    if (slot.candidate && !slot.retiring) {
        slot.retiring = std::move(slot.candidate);
    }
    if (slot.current && !slot.retiring) {
        slot.retiring = std::move(slot.current);
    }
}

void HeatSystemComputeController::quiescedClear() noexcept {
    generationSlots.clear();
    configuredConfigs.clear();
    pendingConfigs.clear();
    failedComputeHashes.clear();
    serialRuntimes.clear();
    lastSerialRevisions.clear();
}

void HeatSystemComputeController::syncSerialInputs() {
    std::unordered_set<uint64_t> requiredSourceKeys;
    for (const auto& [socketKey, config] : configuredConfigs) {
        for (const auto& [sourceKey, portName] : config.serialPortNamesBySourceKey) {
            requiredSourceKeys.insert(sourceKey);
            auto& runtime = serialRuntimes[sourceKey];
            if (!runtime) {
                runtime = std::make_unique<SerialTemperatureRuntime>();
            }
            runtime->configure(
                config.serialEnabledBySourceKey.at(sourceKey),
                portName,
                config.serialBaudRatesBySourceKey.at(sourceKey));
        }
    }

    for (auto it = serialRuntimes.begin(); it != serialRuntimes.end();) {
        if (requiredSourceKeys.find(it->first) == requiredSourceKeys.end()) {
            it = serialRuntimes.erase(it);
        } else {
            ++it;
        }
    }
    lastSerialRevisions.clear();
}

void HeatSystemComputeController::updateSerialInputs() {
    for (auto& [sourceKey, runtime] : serialRuntimes) {
        if (runtime) runtime->update();
    }
}

void HeatSystemComputeController::publishProbeTelemetry(TemperatureTelemetryReceiver& receiver) const {
    for (const auto& [socketKey, slot] : generationSlots) {
        if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
            continue;
        }
        if (slot.current) {
            slot.current->publishProbeTelemetry(receiver);
        }
    }
}

void HeatSystemComputeController::publishSerialTelemetry(ConnectionTelemetryReceiver& receiver) const {
    for (const auto& [sourceKey, runtime] : serialRuntimes) {
        if (!runtime) {
            continue;
        }
        TelemetryConnection telemetry;
        // Serial source keys are packed NodeSocketKey values: (nodeId << 32) | socketId.
        telemetry.nodeId = static_cast<uint32_t>(sourceKey >> 32);
        telemetry.status = &runtime->status();
        receiver.onConnection(telemetry);
    }
}

std::unique_ptr<HeatSystem> HeatSystemComputeController::buildHeatSystem() {
    std::unique_ptr<HeatSystem> system = std::make_unique<HeatSystem>(
        vulkanDevice,
        memoryAllocator,
        maxFramesInFlight,
        renderCommandPool,
        transferCommandPool);
    if (!system || !system->isInitialized()) {
        std::cerr << "[HeatSystemComputeController] HeatSystem initialization failed" << std::endl;
        return nullptr;
    }
    return system;
}

void HeatSystemComputeController::waitIdle() {
    for (auto& [key, slot] : generationSlots) {
        (void)key;
        if (slot.current) slot.current->waitForSolvesIdle();
        if (slot.retiring) slot.retiring->waitForSolvesIdle();
        if (slot.candidate) slot.candidate->waitForSolvesIdle();
    }
}

void HeatSystemComputeController::releaseWaits() {
    for (auto& [key, slot] : generationSlots) {
        (void)key;
        if (slot.current) slot.current->releasePendingSolveWaitOnHost();
        if (slot.retiring) slot.retiring->releasePendingSolveWaitOnHost();
        if (slot.candidate) slot.candidate->releasePendingSolveWaitOnHost();
    }
}

bool HeatSystemComputeController::buildProduct(uint64_t socketKey, HeatProduct& outProduct) {
    outProduct = {};
    if (socketKey == 0) return false;

    const auto sysIt = generationSlots.find(socketKey);
    if (sysIt == generationSlots.end()) return false;
    HeatSystem* systemPtr = sysIt->second.candidate
        ? sysIt->second.candidate.get()
        : sysIt->second.current.get();
    if (!systemPtr) return false;
    HeatSystem& system = *systemPtr;

    const Config* config = getBuildConfig(socketKey);
    if (!config) return false;

    outProduct.modelRuntimeModelIds.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceBuffers.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceBufferOffsets.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfacePointCounts.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceGradientBuffers.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceGradientBufferOffsets.reserve(config->modelRuntimeModelIds.size());

    for (uint32_t runtimeModelId : config->modelRuntimeModelIds) {
        HeatModelRuntime* model = system.getModelByRuntimeId(runtimeModelId);
        if (!model) continue;

        const VkBuffer surfaceBuffer = model->getSurfaceBuffer();
        const VkBuffer gradientBuffer = model->getSurfaceGradientBuffer();
        if (surfaceBuffer == VK_NULL_HANDLE || gradientBuffer == VK_NULL_HANDLE) {
            outProduct = {};
            return false;
        }

        outProduct.modelRuntimeModelIds.push_back(runtimeModelId);
        outProduct.modelSurfaceBuffers.push_back(surfaceBuffer);
        outProduct.modelSurfaceBufferOffsets.push_back(model->getSurfaceBufferOffset());
        outProduct.modelSurfacePointCounts.push_back(static_cast<uint32_t>(model->getSurfaceVertexCount()));
        outProduct.modelSurfaceGradientBuffers.push_back(gradientBuffer);
        outProduct.modelSurfaceGradientBufferOffsets.push_back(model->getSurfaceGradientBufferOffset());
    }

    if (!outProduct.isValid()) {
        outProduct = {};
        return false;
    }

    outProduct.fluidTemperatureVolumeView = system.getFluidTemperatureVolumeView();
    outProduct.fluidTemperatureVolumeSampler = system.getFluidTemperatureVolumeSampler();
    outProduct.hasFluidVolume = (outProduct.fluidTemperatureVolumeView != VK_NULL_HANDLE);

    HashProduct::seal(outProduct, config->geometryHash, config->computeHash);
    return true;
}

const HeatSystem* HeatSystemComputeController::getSystem(uint64_t socketKey) const {
    if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
        return nullptr;
    }
    const auto systemIt = generationSlots.find(socketKey);
    if (systemIt == generationSlots.end() || !systemIt->second.current) {
        return nullptr;
    }
    return systemIt->second.current.get();
}

HeatSystem* HeatSystemComputeController::getSystem(uint64_t socketKey) {
    if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
        return nullptr;
    }
    const auto systemIt = generationSlots.find(socketKey);
    if (systemIt == generationSlots.end() || !systemIt->second.current) {
        return nullptr;
    }
    return systemIt->second.current.get();
}

const HeatSystemComputeController::Config* HeatSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
}

const HeatSystemComputeController::Config* HeatSystemComputeController::getBuildConfig(uint64_t socketKey) const {
    const auto pendingIt = pendingConfigs.find(socketKey);
    if (pendingIt != pendingConfigs.end()) {
        return &pendingIt->second;
    }
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
}
