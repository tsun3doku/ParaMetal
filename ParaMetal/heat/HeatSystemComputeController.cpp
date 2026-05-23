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
                std::cerr << "[HeatSystemComputeController] addVoronoiModelInput SKIP runtimeModelId=" << runtimeModelId << std::endl;
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

const HeatSystem* HeatSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto systemIt = activeSystems.find(socketKey);
    if (systemIt == activeSystems.end() || !systemIt->second->system) {
        return nullptr;
    }
    return systemIt->second->system.get();
}

const HeatSystemComputeController::Config* HeatSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
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

uint64_t buildComputeHash(const HeatSystemComputeController::Config& config) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.active ? 1u : 0u));
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.paused ? 1u : 0u));
    NodeGraphHash::combinePod(hash, config.contactThermalConductance);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.modelIntrinsicMeshes.size()));
    for (const SupportingHalfedge::IntrinsicMesh& mesh : config.modelIntrinsicMeshes) {
        NodeGraphHash::combinePodVector(hash, mesh.vertices);
        NodeGraphHash::combinePodVector(hash, mesh.indices);
        NodeGraphHash::combinePodVector(hash, mesh.faceIds);
        NodeGraphHash::combinePodVector(hash, mesh.triangles);
    }
    NodeGraphHash::combinePodVector(hash, config.modelRuntimeModelIds);
    NodeGraphHash::combinePodVector(hash, config.supportingHalfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.supportingAngleViews);
    NodeGraphHash::combinePodVector(hash, config.halfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.edgeViews);
    NodeGraphHash::combinePodVector(hash, config.triangleViews);
    NodeGraphHash::combinePodVector(hash, config.lengthViews);
    NodeGraphHash::combinePodVector(hash, config.inputHalfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.inputEdgeViews);
    NodeGraphHash::combinePodVector(hash, config.inputTriangleViews);
    NodeGraphHash::combinePodVector(hash, config.inputLengthViews);

    auto mixPodMap = [&hash](const auto& map) {
        NodeGraphHash::combine(hash, static_cast<uint64_t>(map.size()));
        for (const auto& [id, value] : map) {
            NodeGraphHash::combinePod(hash, id);
            NodeGraphHash::combinePod(hash, value);
        }
    };
    mixPodMap(config.modelTemperatureByRuntimeId);
    mixPodMap(config.modelBoundaryConditions);
    mixPodMap(config.modelFixedTemperatureValues);
    mixPodMap(config.modelDensity);
    mixPodMap(config.modelSpecificHeat);
    mixPodMap(config.modelConductivity);
    mixPodMap(config.modelVoronoiNodeBufferByModelId);
    mixPodMap(config.modelVoronoiNodeBufferOffsetByModelId);
    mixPodMap(config.modelSimNodeBufferByModelId);
    mixPodMap(config.modelSimNodeBufferOffsetByModelId);
    mixPodMap(config.modelSimGMLSInterfaceBufferByModelId);
    mixPodMap(config.modelSimGMLSInterfaceBufferOffsetByModelId);
    mixPodMap(config.voronoiNodeCounts);
    mixPodMap(config.simNodeCounts);
    mixPodMap(config.simGMLSInterfaceCounts);
    mixPodMap(config.modelGMLSSurfaceStencilBufferByModelId);
    mixPodMap(config.modelGMLSSurfaceStencilBufferOffsetByModelId);
    mixPodMap(config.modelGMLSSurfaceWeightBufferByModelId);
    mixPodMap(config.modelGMLSSurfaceWeightBufferOffsetByModelId);
    mixPodMap(config.modelGMLSSurfaceGradientWeightBufferByModelId);
    mixPodMap(config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId);

    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.modelVoronoiToSimByModelId.size()));
    for (const auto& [id, mapping] : config.modelVoronoiToSimByModelId) {
        NodeGraphHash::combinePod(hash, id);
        NodeGraphHash::combinePodVector(hash, mapping);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.modelVoronoiSeedFlagsByModelId.size()));
    for (const auto& [id, flags] : config.modelVoronoiSeedFlagsByModelId) {
        NodeGraphHash::combinePod(hash, id);
        NodeGraphHash::combinePodVector(hash, flags);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.modelVoronoiSeedPositionsByModelId.size()));
    for (const auto& [id, positions] : config.modelVoronoiSeedPositionsByModelId) {
        NodeGraphHash::combinePod(hash, id);
        NodeGraphHash::combinePodVector(hash, positions);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.contactCouplings.size()));
    for (const ContactCoupling& coupling : config.contactCouplings) {
        NodeGraphHash::combine(hash, coupling.modelARuntimeModelId);
        NodeGraphHash::combine(hash, coupling.modelBRuntimeModelId);
        NodeGraphHash::combinePodVector(hash, coupling.modelBTriangleIndices);
        NodeGraphHash::combine(hash, coupling.contactPairCount);
    }
    return hash;
}
