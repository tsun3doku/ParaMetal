#include "VoronoiSystemComputeController.hpp"

#include <iostream>
#include "VoronoiSystem.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

VoronoiSystemComputeController::VoronoiSystemComputeController(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    CommandPool& renderCommandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      renderCommandPool(renderCommandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

std::unique_ptr<VoronoiSystem> VoronoiSystemComputeController::buildVoronoiSystem() {
    auto system = std::make_unique<VoronoiSystem>(
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        maxFramesInFlight,
        renderCommandPool);
    if (!system || !system->isInitialized()) {
        return nullptr;
    }

    return system;
}

void VoronoiSystemComputeController::configure(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto& system = activeSystems[socketKey];
    if (!system) {
        system = buildVoronoiSystem();
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        // Hash match
        return;
    }

    configuredConfigs[socketKey] = config;

    if (system) {
        if (config.isPointDomain) {
            system->clearReceiverGeometry();
            system->setPointGeometry(config.pointPositions);
        } else {
            system->setReceiverGeometry(
                config.receiverNodeModelId,
                config.receiverGeometryPositions,
                config.receiverGeometryTriangleIndices,
                config.receiverIntrinsicMesh,
                config.receiverSurfaceVertices,
                config.receiverIntrinsicTriangleIndices,
                config.receiverRuntimeModelId,
                config.meshModelMatrix);
            system->setSeedPositions(config.pointPositions);
        }
        system->setParams(config.cellSize, config.voxelResolution);
        const bool configured = system->ensureConfigured();
        if (configured) {
            configuredConfigs[socketKey] = config;
        } else {
            configuredConfigs.erase(socketKey);
        }
    }
}

void VoronoiSystemComputeController::disable(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredConfigs.erase(socketKey);
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            it->second->cleanupResources();
            it->second->cleanup();
        }
        activeSystems.erase(it);
    }
}

void VoronoiSystemComputeController::disableAll() {
    configuredConfigs.clear();
    for (auto& [key, system] : activeSystems) {
        (void)key;
        if (system) {
            system->cleanupResources();
            system->cleanup();
        }
    }
    activeSystems.clear();
}

std::vector<VoronoiSystem*> VoronoiSystemComputeController::getActiveSystems() const {
    std::vector<VoronoiSystem*> systems;
    systems.reserve(activeSystems.size());
    for (const auto& [key, system] : activeSystems) {
        (void)key;
        if (system && system->isReady()) {
            systems.push_back(system.get());
        }
    }
    return systems;
}

#if 0
bool VoronoiSystemComputeController::disabledProductExport(uint64_t socketKey, VoronoiProduct& outProduct) const {
    outProduct = {};

    if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
        return false;
    }

    const auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end() || !it->second || !it->second->isReady()) {
        return false;
    }

    const auto& voronoiSystem = it->second;

    outProduct.nodeCount = voronoiSystem->getVoronoiNodeCount();
    outProduct.simNodeCount = voronoiSystem->runtimeRef().getSimNodeCount();

    const VoronoiResources& resources = voronoiSystem->resourcesRef();
    outProduct.mappedVoronoiNodes = nullptr;
    outProduct.nodeBuffer = resources.voronoiNodeBuffer;
    outProduct.nodeBufferOffset = resources.voronoiNodeBufferOffset;
    outProduct.voronoiNeighborBuffer = resources.voronoiNeighborBuffer;
    outProduct.voronoiNeighborBufferOffset = resources.voronoiNeighborBufferOffset;
    outProduct.voronoiNeighborIndicesBuffer = resources.voronoiNeighborIndicesBuffer;
    outProduct.voronoiNeighborIndicesBufferOffset = resources.voronoiNeighborIndicesBufferOffset;
    outProduct.voronoiInterfaceAreasBuffer = resources.voronoiInterfaceAreasBuffer;
    outProduct.voronoiInterfaceAreasBufferOffset = resources.voronoiInterfaceAreasBufferOffset;
    outProduct.voronoiInterfaceNeighborIdsBuffer = resources.voronoiInterfaceNeighborIdsBuffer;
    outProduct.voronoiInterfaceNeighborIdsBufferOffset = resources.voronoiInterfaceNeighborIdsBufferOffset;
    outProduct.voronoiGMLSInterfaceBuffer = resources.voronoiGMLSInterfaceBuffer;
    outProduct.voronoiGMLSInterfaceBufferOffset = resources.voronoiGMLSInterfaceBufferOffset;
    outProduct.simNodeBuffer = voronoiSystem->runtimeRef().getSimNodeBuffer();
    outProduct.simNodeBufferOffset = voronoiSystem->runtimeRef().getSimNodeBufferOffset();
    outProduct.simGMLSInterfaceBuffer = voronoiSystem->runtimeRef().getSimGMLSInterfaceBuffer();
    outProduct.simGMLSInterfaceBufferOffset = voronoiSystem->runtimeRef().getSimGMLSInterfaceBufferOffset();
    outProduct.simGMLSInterfaceCount = voronoiSystem->runtimeRef().getSimGMLSInterfaceCount();
    outProduct.voronoiSeedFlagsBuffer = resources.voronoiSeedFlagsBuffer;
    outProduct.voronoiSeedFlagsBufferOffset = resources.voronoiSeedFlagsBufferOffset;
    outProduct.seedPositionBuffer = resources.seedPositionBuffer;
    outProduct.seedPositionBufferOffset = resources.seedPositionBufferOffset;
    outProduct.occupancyPointBuffer = resources.occupancyPointBuffer;
    outProduct.occupancyPointBufferOffset = resources.occupancyPointBufferOffset;
    outProduct.occupancyPointCount = resources.occupancyPointCount;
    outProduct.voronoiToSim = voronoiSystem->runtimeRef().getVoronoiToSim();
    outProduct.simToVoronoi = voronoiSystem->runtimeRef().getSimToVoronoi();

    const auto& modelRuntimes = voronoiSystem->getModelRuntimes();
    const auto& domainSeedFlags = voronoiSystem->runtimeRef().getSeedFlags();
    const auto& domainSeedPositions = voronoiSystem->runtimeRef().getSeedPositions();

    outProduct.modelRuntimeModelIds.reserve(modelRuntimes.size());
    outProduct.modelCandidateBuffers.reserve(modelRuntimes.size());
    outProduct.modelCandidateBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceStencilBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceStencilBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceWeightBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceWeightBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceGradientWeightBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceGradientWeightBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelSeedFlags.reserve(modelRuntimes.size());
    outProduct.modelSeedPositions.reserve(modelRuntimes.size());

    for (const auto& modelRuntime : modelRuntimes) {
        if (!modelRuntime) {
            continue;
        }
        outProduct.modelRuntimeModelIds.push_back(modelRuntime->getRuntimeModelId());
        outProduct.modelCandidateBuffers.push_back(modelRuntime->getVoronoiCandidateBuffer());
        outProduct.modelCandidateBufferOffsets.push_back(modelRuntime->getVoronoiCandidateBufferOffset());
        outProduct.modelGMLSSurfaceStencilBuffers.push_back(modelRuntime->getGMLSSurfaceStencilBuffer());
        outProduct.modelGMLSSurfaceStencilBufferOffsets.push_back(modelRuntime->getGMLSSurfaceStencilBufferOffset());
        outProduct.modelGMLSSurfaceWeightBuffers.push_back(modelRuntime->getGMLSSurfaceWeightBuffer());
        outProduct.modelGMLSSurfaceWeightBufferOffsets.push_back(modelRuntime->getGMLSSurfaceWeightBufferOffset());
        outProduct.modelGMLSSurfaceWeightCounts.push_back(modelRuntime->getGMLSSurfaceWeightCount());
        outProduct.modelGMLSSurfaceGradientWeightBuffers.push_back(modelRuntime->getGMLSSurfaceGradientWeightBuffer());
        outProduct.modelGMLSSurfaceGradientWeightBufferOffsets.push_back(modelRuntime->getGMLSSurfaceGradientWeightBufferOffset());
        outProduct.modelGMLSSurfaceGradientWeightCounts.push_back(modelRuntime->getGMLSSurfaceGradientWeightCount());
        outProduct.modelSeedFlags.push_back(domainSeedFlags);
        std::vector<glm::vec3> seedPositions;
        seedPositions.reserve(domainSeedPositions.size());
        for (const glm::vec4& pos : domainSeedPositions) {
            seedPositions.push_back(glm::vec3(pos));
        }
        outProduct.modelSeedPositions.push_back(std::move(seedPositions));
    }

    outProduct.productHash = buildProductHash(outProduct);

    return outProduct.isValid();
}
#endif

const VoronoiSystem* VoronoiSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end() || !it->second || !it->second->isReady()) {
        return nullptr;
    }
    return it->second.get();
}

const VoronoiSystemComputeController::Config* VoronoiSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}
