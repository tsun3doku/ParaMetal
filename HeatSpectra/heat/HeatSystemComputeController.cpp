#include "HeatSystemComputeController.hpp"

#include <iostream>
#include "HeatSystem.hpp"
#include "heat/HeatModelRuntime.hpp"
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
    if (!config.modelVoronoiNodeCountByModelId.empty()) {
        for (const auto& [runtimeModelId, nodeCount] : config.modelVoronoiNodeCountByModelId) {
            const auto nodesIt = config.modelVoronoiNodesByModelId.find(runtimeModelId);
            const auto nodeBufferIt = config.modelVoronoiNodeBufferByModelId.find(runtimeModelId);
            const auto nodeBufferOffsetIt = config.modelVoronoiNodeBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsInterfaceIt = config.modelGMLSInterfaceBufferByModelId.find(runtimeModelId);
            const auto gmlsInterfaceOffsetIt = config.modelGMLSInterfaceBufferOffsetByModelId.find(runtimeModelId);
            const auto seedFlagsBufferIt = config.modelSeedFlagsBufferByModelId.find(runtimeModelId);
            const auto seedFlagsBufferOffsetIt = config.modelSeedFlagsBufferOffsetByModelId.find(runtimeModelId);
            const auto countIt = config.modelVoronoiNodeCountByModelId.find(runtimeModelId);
            const auto gmlsStencilIt = config.modelGMLSSurfaceStencilBufferByModelId.find(runtimeModelId);
            const auto gmlsStencilOffsetIt = config.modelGMLSSurfaceStencilBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsWeightIt = config.modelGMLSSurfaceWeightBufferByModelId.find(runtimeModelId);
            const auto gmlsWeightOffsetIt = config.modelGMLSSurfaceWeightBufferOffsetByModelId.find(runtimeModelId);
            const auto gmlsGradientIt = config.modelGMLSSurfaceGradientWeightBufferByModelId.find(runtimeModelId);
            const auto gmlsGradientOffsetIt = config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId.find(runtimeModelId);
            const auto seedFlagsIt = config.modelVoronoiSeedFlagsByModelId.find(runtimeModelId);
            const auto seedPositionsIt = config.modelVoronoiSeedPositionsByModelId.find(runtimeModelId);
            
            if (countIt == config.modelVoronoiNodeCountByModelId.end() ||
                nodeBufferIt == config.modelVoronoiNodeBufferByModelId.end() ||
                nodeBufferOffsetIt == config.modelVoronoiNodeBufferOffsetByModelId.end() ||
                gmlsInterfaceIt == config.modelGMLSInterfaceBufferByModelId.end() ||
                gmlsInterfaceOffsetIt == config.modelGMLSInterfaceBufferOffsetByModelId.end() ||
                seedFlagsBufferIt == config.modelSeedFlagsBufferByModelId.end() ||
                seedFlagsBufferOffsetIt == config.modelSeedFlagsBufferOffsetByModelId.end() ||
                seedFlagsIt == config.modelVoronoiSeedFlagsByModelId.end() ||
                seedPositionsIt == config.modelVoronoiSeedPositionsByModelId.end()) {
                std::cerr << "[HeatSystemComputeController] addVoronoiModelInput SKIP runtimeModelId=" << runtimeModelId << std::endl;
                continue;
            }

            system.addVoronoiModelInput(
                runtimeModelId,
                nodesIt->second,
                countIt->second,
                nodeBufferIt->second,
                nodeBufferOffsetIt->second,
                gmlsInterfaceIt->second,
                gmlsInterfaceOffsetIt->second,
                seedFlagsBufferIt->second,
                seedFlagsBufferOffsetIt->second,
                (gmlsStencilIt != config.modelGMLSSurfaceStencilBufferByModelId.end()) ? gmlsStencilIt->second : VK_NULL_HANDLE,
                (gmlsStencilOffsetIt != config.modelGMLSSurfaceStencilBufferOffsetByModelId.end()) ? gmlsStencilOffsetIt->second : 0,
                (gmlsWeightIt != config.modelGMLSSurfaceWeightBufferByModelId.end()) ? gmlsWeightIt->second : VK_NULL_HANDLE,
                (gmlsWeightOffsetIt != config.modelGMLSSurfaceWeightBufferOffsetByModelId.end()) ? gmlsWeightOffsetIt->second : 0,
                (gmlsGradientIt != config.modelGMLSSurfaceGradientWeightBufferByModelId.end()) ? gmlsGradientIt->second : VK_NULL_HANDLE,
                (gmlsGradientOffsetIt != config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId.end()) ? gmlsGradientOffsetIt->second : 0,
                seedFlagsIt->second,
                seedPositionsIt->second);
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
    system.setParams(config.contactThermalConductance);
    system.setContactCouplings(config.contactCouplings);
}

void HeatSystemComputeController::configure(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end()) {
        auto newInstance = std::make_unique<SystemInstance>();
        newInstance->system = buildHeatSystem(newInstance->resources);
        it = activeSystems.emplace(socketKey, std::move(newInstance)).first;
    }

    auto& instance = it->second;

    if (instance->system) {
        instance->system->setActive(config.active);
        instance->system->setIsPaused(config.active && config.paused);

        if (config.resetRequested) {
            std::cerr << "[HeatSystemComputeController] explicit reset socketKey=" << socketKey << std::endl;
            instance->system->resetHeatState(config.modelTemperatureByRuntimeId);
        }
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end()) {
        configIt->second.paused = config.paused;
        configIt->second.resetRequested = config.resetRequested;
        if (configIt->second.computeHash == config.computeHash) {
            return;
        }
    }

    configuredConfigs[socketKey] = config;

    if (instance->system) {
        configureHeatSystem(*instance->system, config);
        instance->system->ensureConfigured();
        instance->system->resetHeatState(config.modelTemperatureByRuntimeId);
    }
}

void HeatSystemComputeController::disable(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredConfigs.erase(socketKey);
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second->system) {
            it->second->system->setIsPaused(false);
            it->second->system->setActive(false);
            it->second->system->cleanupResources();
            it->second->system->cleanup();
        }
        activeSystems.erase(it);
    }
}

void HeatSystemComputeController::disableAll() {
    configuredConfigs.clear();
    for (auto& [key, instance] : activeSystems) {
        if (instance->system) {
            instance->system->setIsPaused(false);
            instance->system->setActive(false);
            instance->system->cleanupResources();
            instance->system->cleanup();
        }
    }
    activeSystems.clear();
}

bool HeatSystemComputeController::isAnyHeatSystemActive() const {
    for (const auto& [key, config] : configuredConfigs) {
        auto systemIt = activeSystems.find(key);
        if (config.active && systemIt != activeSystems.end() && systemIt->second->system) {
            return true;
        }
    }
    return false;
}

bool HeatSystemComputeController::isAnyHeatSystemPaused() const {
    for (const auto& [key, config] : configuredConfigs) {
        auto systemIt = activeSystems.find(key);
        if (config.active && config.paused && systemIt != activeSystems.end() && systemIt->second->system) {
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
           systemIt->second->system &&
           configIt->second.active;
}

bool HeatSystemComputeController::isHeatSystemPaused(uint64_t socketKey) const {
    auto configIt = configuredConfigs.find(socketKey);
    auto systemIt = activeSystems.find(socketKey);
    return configIt != configuredConfigs.end() &&
           systemIt != activeSystems.end() &&
           systemIt->second->system &&
           configIt->second.active &&
           configIt->second.paused;
}

std::unique_ptr<HeatSystem> HeatSystemComputeController::buildHeatSystem(HeatSystemResources& heatSystemResources) {
    std::unique_ptr<HeatSystem> system = std::make_unique<HeatSystem>(
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        heatSystemResources,
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
    for (const auto& [key, instance] : activeSystems) {
        if (instance->system) {
            systems.push_back(instance->system.get());
        }
    }
    return systems;
}

bool HeatSystemComputeController::exportProduct(uint64_t socketKey, HeatProduct& outProduct) const {
    outProduct = {};

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt == configuredConfigs.end()) {
        return false;
    }

    const auto systemIt = activeSystems.find(socketKey);
    if (systemIt == activeSystems.end() || !systemIt->second->system) {
        return false;
    }
    const HeatSystem& system = *systemIt->second->system;
    outProduct.active = system.getIsActive();
    outProduct.paused = system.getIsPaused();

    const auto& activeModels = system.getActiveModels();
    outProduct.modelRuntimeModelIds.reserve(activeModels.size());
    outProduct.modelSurfaceBuffers.reserve(activeModels.size());
    outProduct.modelSurfaceBufferOffsets.reserve(activeModels.size());
    outProduct.modelSurfacePointCounts.reserve(activeModels.size());
    outProduct.modelSurfaceGradientBuffers.reserve(activeModels.size());
    outProduct.modelSurfaceGradientBufferOffsets.reserve(activeModels.size());
    for (uint32_t runtimeModelId : configIt->second.modelRuntimeModelIds) {
        const HeatModelRuntime* heatModel = system.getModelByRuntimeId(runtimeModelId);
        if (!heatModel ||
            runtimeModelId == 0 ||
            heatModel->getSurfaceBuffer() == VK_NULL_HANDLE ||
            heatModel->getIntrinsicVertexCount() == 0) {
            continue;
        }

        outProduct.modelRuntimeModelIds.push_back(runtimeModelId);
        outProduct.modelSurfaceBuffers.push_back(heatModel->getSurfaceBuffer());
        outProduct.modelSurfaceBufferOffsets.push_back(heatModel->getSurfaceBufferOffset());
        outProduct.modelSurfacePointCounts.push_back(static_cast<uint32_t>(heatModel->getIntrinsicVertexCount()));
        outProduct.modelSurfaceGradientBuffers.push_back(heatModel->getSurfaceGradientBuffer());
        outProduct.modelSurfaceGradientBufferOffsets.push_back(heatModel->getSurfaceGradientBufferOffset());
    }

    outProduct.productHash = buildProductHash(outProduct);
    return outProduct.isValid();
}

void HeatSystemComputeController::destroyHeatSystem(uint64_t socketKey) {
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second->system) {
            it->second->system->cleanupResources();
            it->second->system->cleanup();
        }
        activeSystems.erase(it);
    }
}
