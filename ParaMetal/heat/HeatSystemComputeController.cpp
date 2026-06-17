#include "HeatSystemComputeController.hpp"

#include <iostream>
#include "HeatSystem.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanDevice.hpp"

HeatSystemComputeController::HeatSystemComputeController(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, ModelRegistry& resourceManager,
    CommandPool& renderCommandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      renderCommandPool(renderCommandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

void HeatSystemComputeController::configureHeatSystem(HeatSystem& system, const Config& config) {
    system.clearVoronoiInputs();
    if (!config.simNodeCounts.empty()) {
        for (const auto& [runtimeModelId, simNodeCount] : config.simNodeCounts) {
            (void)simNodeCount;
            const auto nodesIt = config.modelVoronoiNodesByModelId.find(runtimeModelId);
            const auto voronoiNodeBufferIt = config.modelVoronoiNodeBufferByModelId.find(runtimeModelId);
            const auto voronoiNodeBufferOffsetIt = config.modelVoronoiNodeBufferOffsetByModelId.find(runtimeModelId);
            const auto simNodeBufferIt = config.modelSimNodeBufferByModelId.find(runtimeModelId);
            const auto simNodeBufferOffsetIt = config.modelSimNodeBufferOffsetByModelId.find(runtimeModelId);
            const auto simGmlsInterfaceIt = config.modelSimGMLSInterfaceBufferByModelId.find(runtimeModelId);
            const auto simGmlsInterfaceOffsetIt = config.modelSimGMLSInterfaceBufferOffsetByModelId.find(runtimeModelId);
            const auto simGmlsInterfaceCountIt = config.simGMLSInterfaceCounts.find(runtimeModelId);
            const auto voronoiCountIt = config.voronoiNodeCounts.find(runtimeModelId);
            const auto simCountIt = config.simNodeCounts.find(runtimeModelId);
            const auto gmlsStencilIt = config.modelGMLSSurfaceStencilBufferByModelId.find(runtimeModelId);
            const auto gmlsStencilOffsetIt = config.modelGMLSSurfaceStencilBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsWeightIt = config.modelGMLSSurfaceWeightBufferByModelId.find(runtimeModelId);
            const auto gmlsWeightOffsetIt = config.modelGMLSSurfaceWeightBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsWeightCountIt = config.modelGMLSSurfaceWeightCountByModelId.find(runtimeModelId);
            const auto gmlsGradientIt = config.modelGMLSSurfaceGradientWeightBufferByModelId.find(runtimeModelId);
            const auto gmlsGradientOffsetIt = config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsGradientCountIt = config.modelGMLSSurfaceGradientWeightCountByModelId.find(runtimeModelId);
            const auto seedFlagsIt = config.modelVoronoiSeedFlagsByModelId.find(runtimeModelId);
            const auto seedPositionsIt = config.modelVoronoiSeedPositionsByModelId.find(runtimeModelId);
            const auto voronoiToSimIt = config.modelVoronoiToSimByModelId.find(runtimeModelId);
            
            if (voronoiCountIt == config.voronoiNodeCounts.end() ||
                simCountIt == config.simNodeCounts.end() ||
                voronoiNodeBufferIt == config.modelVoronoiNodeBufferByModelId.end() ||
                voronoiNodeBufferOffsetIt == config.modelVoronoiNodeBufferOffsetByModelId.end() ||
                simNodeBufferIt == config.modelSimNodeBufferByModelId.end() ||
                simNodeBufferOffsetIt == config.modelSimNodeBufferOffsetByModelId.end() ||
                simGmlsInterfaceIt == config.modelSimGMLSInterfaceBufferByModelId.end() ||
                simGmlsInterfaceOffsetIt == config.modelSimGMLSInterfaceBufferOffsetByModelId.end() ||
                simGmlsInterfaceCountIt == config.simGMLSInterfaceCounts.end() ||
                seedFlagsIt == config.modelVoronoiSeedFlagsByModelId.end() ||
                seedPositionsIt == config.modelVoronoiSeedPositionsByModelId.end() ||
                voronoiToSimIt == config.modelVoronoiToSimByModelId.end()) {
                continue;
            }

            system.addVoronoiModelInput(
                runtimeModelId,
                nodesIt->second,
                voronoiCountIt->second,
                voronoiNodeBufferIt->second,
                voronoiNodeBufferOffsetIt->second,
                simCountIt->second,
                simNodeBufferIt->second,
                simNodeBufferOffsetIt->second,
                simGmlsInterfaceIt->second,
                simGmlsInterfaceOffsetIt->second,
                simGmlsInterfaceCountIt->second,
                (gmlsStencilIt != config.modelGMLSSurfaceStencilBufferByModelId.end()) ? gmlsStencilIt->second : VK_NULL_HANDLE,
                (gmlsStencilOffsetIt != config.modelGMLSSurfaceStencilBufferOffsetByModelId.end()) ? gmlsStencilOffsetIt->second : 0,
                (gmlsWeightIt != config.modelGMLSSurfaceWeightBufferByModelId.end()) ? gmlsWeightIt->second : VK_NULL_HANDLE,
                (gmlsWeightOffsetIt != config.modelGMLSSurfaceWeightBufferOffsetByModelId.end()) ? gmlsWeightOffsetIt->second : 0,
                (gmlsWeightCountIt != config.modelGMLSSurfaceWeightCountByModelId.end()) ? gmlsWeightCountIt->second : 0,
                (gmlsGradientIt != config.modelGMLSSurfaceGradientWeightBufferByModelId.end()) ? gmlsGradientIt->second : VK_NULL_HANDLE,
                (gmlsGradientOffsetIt != config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId.end()) ? gmlsGradientOffsetIt->second : 0,
                (gmlsGradientCountIt != config.modelGMLSSurfaceGradientWeightCountByModelId.end()) ? gmlsGradientCountIt->second : 0,
                seedFlagsIt->second,
                seedPositionsIt->second,
                voronoiToSimIt->second);
        }
    }
    system.setHeatModels(
        config.modelIntrinsicMeshes,
        config.modelRuntimeModelIds,
        config.modelTemperatureByRuntimeId,
        config.modelBoundaryConditions,
        config.modelFixedTemperatureValues,
        config.modelDensity,
        config.modelSpecificHeat,
        config.modelConductivity);
    system.setParams(config.contactThermalConductance, config.simulationDuration);
    system.setContactCouplings(config.contactCouplings);
}

void HeatSystemComputeController::applyRuntimeState(HeatSystem& system, const Config& config) {
    system.setActive(config.active);
    system.setPlaybackState(config.paused, config.resetCounter);
    system.setRewindFrame(config.rewindFrame);
}

void HeatSystemComputeController::configure(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end()) {
        auto system = buildHeatSystem();
        it = activeSystems.emplace(socketKey, std::move(system)).first;
    }

    auto& system = it->second;
    if (!system) {
        return;
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        applyRuntimeState(*system, config);
        return;
    }

    configureHeatSystem(*system, config);
    const bool configured = system->ensureConfigured();
    if (configured) {
        configuredConfigs[socketKey] = config;
    } else {
        configuredConfigs.erase(socketKey);
    }

    // Runtime state: pushed after structural path so playback instances exist
    applyRuntimeState(*system, config);
}

void HeatSystemComputeController::disable(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredConfigs.erase(socketKey);
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            it->second->setPlaybackState(false, 0);
            it->second->setActive(false);
            it->second->cleanupResources();
            it->second->cleanup();
        }
        activeSystems.erase(it);
    }
}

void HeatSystemComputeController::disableAll() {
    configuredConfigs.clear();
    for (auto& [key, system] : activeSystems) {
        if (system) {
            system->setPlaybackState(false, 0);
            system->setActive(false);
            system->cleanupResources();
            system->cleanup();
        }
    }
    activeSystems.clear();
}

bool HeatSystemComputeController::isAnyHeatSystemActive() const {
    for (const auto& [key, config] : configuredConfigs) {
        auto systemIt = activeSystems.find(key);
        if (config.active && systemIt != activeSystems.end() && systemIt->second) {
            return true;
        }
    }
    return false;
}

bool HeatSystemComputeController::isAnyHeatSystemPaused() const {
    for (const auto& [key, system] : activeSystems) {
        if (system && system->getIsActive() && system->getIsPaused()) {
            return true;
        }
    }
    return false;
}

bool HeatSystemComputeController::isHeatSystemActive(uint64_t socketKey) const {
    auto configIt = configuredConfigs.find(socketKey);
    auto systemIt = activeSystems.find(socketKey);
    return configIt != configuredConfigs.end() &&
           systemIt != activeSystems.end() &&
           systemIt->second &&
           configIt->second.active;
}

bool HeatSystemComputeController::isHeatSystemPaused(uint64_t socketKey) const {
    auto configIt = configuredConfigs.find(socketKey);
    auto systemIt = activeSystems.find(socketKey);
    return configIt != configuredConfigs.end() &&
           systemIt != activeSystems.end() &&
           systemIt->second &&
           configIt->second.active &&
           systemIt->second->getIsPaused();
}

std::unique_ptr<HeatSystem> HeatSystemComputeController::buildHeatSystem() {
    std::unique_ptr<HeatSystem> system = std::make_unique<HeatSystem>(
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        maxFramesInFlight,
        renderCommandPool);
    if (!system || !system->isInitialized()) {
        std::cerr << "[HeatSystemComputeController] HeatSystem initialization failed" << std::endl;
        return nullptr;
    }
    return system;
}

std::vector<ComputePass*> HeatSystemComputeController::getActiveSystems() const {
    std::vector<ComputePass*> systems;
    systems.reserve(activeSystems.size());
    for (const auto& [key, system] : activeSystems) {
        if (system) {
            systems.push_back(system.get());
        }
    }
    return systems;
}

const HeatSystem* HeatSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto systemIt = activeSystems.find(socketKey);
    if (systemIt == activeSystems.end() || !systemIt->second) {
        return nullptr;
    }
    return systemIt->second.get();
}

const HeatSystemComputeController::Config* HeatSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
}

void HeatSystemComputeController::destroyHeatSystem(uint64_t socketKey) {
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            it->second->cleanupResources();
            it->second->cleanup();
        }
        activeSystems.erase(it);
    }
}

