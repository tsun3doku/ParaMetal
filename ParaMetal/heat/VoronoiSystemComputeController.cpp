#include "VoronoiSystemComputeController.hpp"

#include <iostream>
#include "VoronoiSystem.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "hash/HashProduct.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiDomainRuntime.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"

VoronoiSystemComputeController::VoronoiSystemComputeController(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    CommandPool& commandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

std::unique_ptr<VoronoiSystem> VoronoiSystemComputeController::buildVoronoiSystem() {
    auto system = std::make_unique<VoronoiSystem>(
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        maxFramesInFlight,
        commandPool);
    if (!system || !system->isInitialized()) {
        std::cerr << "[VoronoiSystemComputeController] VoronoiSystem initialization failed" << std::endl;
        return nullptr;
    }

    return system;
}

void VoronoiSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end()) {
        auto system = buildVoronoiSystem();
        it = systemsBySocket.emplace(socketKey, std::move(system)).first;
    }

    auto& system = it->second;
    if (!system) {
        return;
    }

    const auto configIt = configuredConfigs.find(socketKey);
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        return;
    }

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

void VoronoiSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredConfigs.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            vkDeviceWaitIdle(vulkanDevice.getDevice());
            it->second->cleanupResources();
            it->second->cleanup();
        }
        systemsBySocket.erase(it);
    }
}

void VoronoiSystemComputeController::disableAll() {
    configuredConfigs.clear();
    if (!systemsBySocket.empty()) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
    }

    for (auto& [key, system] : systemsBySocket) {
        (void)key;
        if (system) {
            system->cleanupResources();
            system->cleanup();
        }
    }
    systemsBySocket.clear();
}

std::vector<VoronoiSystem*> VoronoiSystemComputeController::getActiveSystems() const {
    std::vector<VoronoiSystem*> systems;
    systems.reserve(systemsBySocket.size());
    for (const auto& [key, system] : systemsBySocket) {
        (void)key;
        if (system && system->isReady()) {
            systems.push_back(system.get());
        }
    }
    return systems;
}


const VoronoiSystem* VoronoiSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end() || !it->second || !it->second->isReady()) {
        return nullptr;
    }
    return it->second.get();
}

bool VoronoiSystemComputeController::buildProduct(uint64_t socketKey, VoronoiProduct& outProduct) const {
    outProduct = {};
    const VoronoiSystem* voronoiSystem = getSystem(socketKey);
    if (!voronoiSystem) {
        return false;
    }
    if (!getConfig(socketKey)) {
        return false;
    }

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
    outProduct.simNodeBufferSize = voronoiSystem->runtimeRef().getSimNodeBufferSize();
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
    outProduct.simNodeVolumes = voronoiSystem->runtimeRef().getSimNodeVolumes();

    const VoronoiDomainRuntime* domainRuntime = voronoiSystem->getDomainRuntime();
    if (!domainRuntime) {
        return false;
    }
    const auto& domainSeedFlags = voronoiSystem->runtimeRef().getSeedFlags();
    const auto& domainSeedPositions = voronoiSystem->runtimeRef().getSeedPositions();

    outProduct.isPointDomain = domainRuntime->isPointDomain();
    outProduct.candidateBuffer = domainRuntime->getCandidateBuffer();
    outProduct.candidateBufferOffset = domainRuntime->getCandidateBufferOffset();

    if (!domainRuntime->isPointDomain()) {
        const VoronoiModelRuntime* modelRuntime = static_cast<const VoronoiModelRuntime*>(domainRuntime);
        outProduct.runtimeModelId = modelRuntime->getRuntimeModelId();
        outProduct.gmlsSurfaceStencilBuffer = modelRuntime->getGMLSSurfaceStencilBuffer();
        outProduct.gmlsSurfaceStencilBufferOffset = modelRuntime->getGMLSSurfaceStencilBufferOffset();
        outProduct.gmlsSurfaceWeightBuffer = modelRuntime->getGMLSSurfaceWeightBuffer();
        outProduct.gmlsSurfaceWeightBufferOffset = modelRuntime->getGMLSSurfaceWeightBufferOffset();
        outProduct.gmlsSurfaceWeightCount = modelRuntime->getGMLSSurfaceWeightCount();
        outProduct.gmlsSurfaceGradientWeightBuffer = modelRuntime->getGMLSSurfaceGradientWeightBuffer();
        outProduct.gmlsSurfaceGradientWeightBufferOffset = modelRuntime->getGMLSSurfaceGradientWeightBufferOffset();
        outProduct.gmlsSurfaceGradientWeightCount = modelRuntime->getGMLSSurfaceGradientWeightCount();
    } else {
        outProduct.runtimeModelId = 0;
    }

    outProduct.seedFlags = domainSeedFlags;
    outProduct.seedPositions.reserve(domainSeedPositions.size());
    for (const glm::vec4& pos : domainSeedPositions) {
        outProduct.seedPositions.push_back(glm::vec3(pos));
    }

    HashProduct::seal(outProduct);
    return outProduct.isValid();
}

const VoronoiSystemComputeController::Config* VoronoiSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}
