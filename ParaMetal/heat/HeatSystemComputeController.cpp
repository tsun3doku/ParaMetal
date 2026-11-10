#include "HeatSystemComputeController.hpp"

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

void HeatSystemComputeController::configureHeatSystem(HeatSystem& system, const Config& config) {
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
    if (config.domainVoronoiProduct.isValid()) {
        system.setGlobalVoronoiInput(config.domainVoronoiProduct, config.modelLocalToWorldByModelId);
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

    configureHeatSystem(*candidate, config);
    if (!candidate->ensureConfigured()) {
        generationSlots[socketKey].retiring = std::move(candidate);
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }
    candidate->syncHeatProbes(config.heatProbesBySourceKey);

    GenerationSlot& slot = generationSlots[socketKey];
    configuredConfigs.try_emplace(socketKey);
    pendingConfigs[socketKey] = config;
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

bool HeatSystemComputeController::getSerialTemperatureStatus(
    uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const {
    const auto it = serialRuntimes.find(sourceKey);
    if (it == serialRuntimes.end() || !it->second) return false;
    outStatus = it->second->status();
    return true;
}

bool HeatSystemComputeController::getHeatProbeTemperatures(uint64_t sourceKey, std::vector<float>& outTemperaturesC) const {
    for (const auto& [socketKey, slot] : generationSlots) {
        if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
            continue;
        }
        if (slot.current && slot.current->getHeatProbeTemperatures(sourceKey, outTemperaturesC)) {
            return true;
        }
    }
    return false;
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

    HashProduct::seal(outProduct);
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
