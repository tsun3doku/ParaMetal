#include "HeatSystemComputeController.hpp"

#include <iostream>
#include "HeatSystem.hpp"
#include "hash/HashProduct.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanDevice.hpp"

HeatSystemComputeController::HeatSystemComputeController(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, ModelRegistry& resourceManager,
    CommandPool& renderCommandPool,
    CommandPool& transferCommandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      renderCommandPool(renderCommandPool),
      transferCommandPool(transferCommandPool),
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
            const auto simNodeBufferSizeIt = config.modelSimNodeBufferSizeByModelId.find(runtimeModelId);
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
            const auto simVolumesIt = config.modelSimNodeVolumesByModelId.find(runtimeModelId);
            const auto voronoiToSimIt = config.modelVoronoiToSimByModelId.find(runtimeModelId);
            
            if (voronoiCountIt == config.voronoiNodeCounts.end() ||
                simCountIt == config.simNodeCounts.end() ||
                voronoiNodeBufferIt == config.modelVoronoiNodeBufferByModelId.end() ||
                voronoiNodeBufferOffsetIt == config.modelVoronoiNodeBufferOffsetByModelId.end() ||
                simNodeBufferIt == config.modelSimNodeBufferByModelId.end() ||
                simNodeBufferOffsetIt == config.modelSimNodeBufferOffsetByModelId.end() ||
                simNodeBufferSizeIt == config.modelSimNodeBufferSizeByModelId.end() ||
                simGmlsInterfaceIt == config.modelSimGMLSInterfaceBufferByModelId.end() ||
                simGmlsInterfaceOffsetIt == config.modelSimGMLSInterfaceBufferOffsetByModelId.end() ||
                simGmlsInterfaceCountIt == config.simGMLSInterfaceCounts.end() ||
                seedFlagsIt == config.modelVoronoiSeedFlagsByModelId.end() ||
                seedPositionsIt == config.modelVoronoiSeedPositionsByModelId.end() ||
                simVolumesIt == config.modelSimNodeVolumesByModelId.end() ||
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
                simNodeBufferSizeIt->second,
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
                simVolumesIt->second,
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

void HeatSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end()) {
        auto system = buildHeatSystem();
        it = systemsBySocket.emplace(socketKey, std::move(system)).first;
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

    applyRuntimeState(*system, config);
}

void HeatSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredConfigs.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            it->second->setPlaybackState(false, 0);
            it->second->setActive(false);
            vkDeviceWaitIdle(vulkanDevice.getDevice());
            it->second->cleanup();
        }
        systemsBySocket.erase(it);
    }
}

void HeatSystemComputeController::disableAll() {
    configuredConfigs.clear();
    if (!systemsBySocket.empty()) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
    }
    for (auto& [key, system] : systemsBySocket) {
        if (system) {
            system->setPlaybackState(false, 0);
            system->setActive(false);
            system->cleanup();
        }
    }
    systemsBySocket.clear();
}

bool HeatSystemComputeController::isAnyHeatSystemActive() const {
    for (const auto& [key, config] : configuredConfigs) {
        auto systemIt = systemsBySocket.find(key);
        if (config.active && systemIt != systemsBySocket.end() && systemIt->second) {
            return true;
        }
    }
    return false;
}

bool HeatSystemComputeController::isAnyHeatSystemPaused() const {
    for (const auto& [key, system] : systemsBySocket) {
        if (system && system->getIsActive() && system->getIsPaused()) {
            return true;
        }
    }
    return false;
}

std::unique_ptr<HeatSystem> HeatSystemComputeController::buildHeatSystem() {
    std::unique_ptr<HeatSystem> system = std::make_unique<HeatSystem>(
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        maxFramesInFlight,
        renderCommandPool,
        transferCommandPool);
    if (!system || !system->isInitialized()) {
        std::cerr << "[HeatSystemComputeController] HeatSystem initialization failed" << std::endl;
        return nullptr;
    }
    return system;
}

std::vector<ComputePass*> HeatSystemComputeController::getActiveSystems() const {
    std::vector<ComputePass*> systems;
    systems.reserve(systemsBySocket.size());
    for (const auto& [key, system] : systemsBySocket) {
        if (system && system->hasDispatchableComputeWork()) {
            systems.push_back(system.get());
        }
    }
    return systems;
}

bool HeatSystemComputeController::buildProduct(uint64_t socketKey, HeatProduct& outProduct) {
    outProduct = {};
    if (socketKey == 0) return false;

    auto sysIt = systemsBySocket.find(socketKey);
    if (sysIt == systemsBySocket.end() || !sysIt->second) return false;
    HeatSystem& system = *sysIt->second;

    const Config* config = getConfig(socketKey);
    if (!config) return false;

    const auto freeProduct = [&]() {
        for (size_t j = 0; j < outProduct.modelSurfaceBuffers.size(); ++j) {
            if (outProduct.modelSurfaceBuffers[j] != VK_NULL_HANDLE)
                memoryAllocator.free(outProduct.modelSurfaceBuffers[j],
                    j < outProduct.modelSurfaceBufferOffsets.size() ? outProduct.modelSurfaceBufferOffsets[j] : 0);
        }
        for (size_t j = 0; j < outProduct.modelSurfaceGradientBuffers.size(); ++j) {
            if (outProduct.modelSurfaceGradientBuffers[j] != VK_NULL_HANDLE)
                memoryAllocator.free(outProduct.modelSurfaceGradientBuffers[j],
                    j < outProduct.modelSurfaceGradientBufferOffsets.size() ? outProduct.modelSurfaceGradientBufferOffsets[j] : 0);
        }
        outProduct = {};
    };

    outProduct.modelRuntimeModelIds.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceBuffers.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceBufferOffsets.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfacePointCounts.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceGradientBuffers.reserve(config->modelRuntimeModelIds.size());
    outProduct.modelSurfaceGradientBufferOffsets.reserve(config->modelRuntimeModelIds.size());

    for (uint32_t runtimeModelId : config->modelRuntimeModelIds) {
        HeatModelRuntime* model = system.getModelByRuntimeId(runtimeModelId);
        if (!model || runtimeModelId == 0) continue;

        if (!model->appendProduct(outProduct)) {
            freeProduct();
            return false;
        }
        outProduct.modelRuntimeModelIds.push_back(runtimeModelId);
    }

    if (!outProduct.isValid()) {
        freeProduct();
        return false;
    }

    if (!system.setupDescriptors(
            outProduct.modelSurfaceBuffers,
            outProduct.modelSurfaceBufferOffsets,
            outProduct.modelSurfaceGradientBuffers,
            outProduct.modelSurfaceGradientBufferOffsets)) {
        freeProduct();
        return false;
    }

    HashProduct::seal(outProduct);
    return true;
}

const HeatSystem* HeatSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto systemIt = systemsBySocket.find(socketKey);
    if (systemIt == systemsBySocket.end() || !systemIt->second) {
        return nullptr;
    }
    return systemIt->second.get();
}

const HeatSystemComputeController::Config* HeatSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
}


