#include "VoronoiSystemComputeController.hpp"

#include "voronoi/VoronoiSystem.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "hash/HashProduct.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

VoronoiSystemComputeController::VoronoiSystemComputeController(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool) {
}

bool VoronoiSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    const auto configIt = configuredHashes.find(socketKey);
    if (configIt != configuredHashes.end() && configIt->second == config.computeHash) {
        return true;
    }

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    // Build replacement generation off to the side while active generation remains live
    auto candidate = std::make_unique<VoronoiSystem>(vulkanDevice, memoryAllocator, commandPool);
    candidate->setParams(config.cellSize, config.voxelResolution);
    candidate->setContactRange(config.contactRange);
    candidate->setSeedPositions(config.pointPositions, config.pointDomainCorners);
    candidate->setGlobalGeometry(
        config.globalRemeshRuntimeModelIds,
        config.globalRemeshPositions,
        config.globalRemeshTriangleIndices,
        config.globalRemeshSurfacePositions,
        config.globalRemeshSurfaceTriangleIndices,
        config.volumePadding);

    if (!candidate->ensureConfigured()) {
        // Retain failed candidate to avoid destroying resources with in-flight GPU work
        retainedSystems.push_back(std::move(candidate));
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            retainedSystems.push_back(std::move(it->second));
        }
        it->second = std::move(candidate);
    } else {
        systemsBySocket.emplace(socketKey, std::move(candidate));
    }

    configuredHashes[socketKey] = config.computeHash;
    failedComputeHashes.erase(socketKey);
    return true;
}

void VoronoiSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredHashes.erase(socketKey);
    failedComputeHashes.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            retainedSystems.push_back(std::move(it->second));
        }
        systemsBySocket.erase(it);
    }
}

void VoronoiSystemComputeController::disableAll() {
    configuredHashes.clear();
    failedComputeHashes.clear();
    for (auto& [socketKey, system] : systemsBySocket) {
        (void)socketKey;
        if (system) {
            retainedSystems.push_back(std::move(system));
        }
    }
    systemsBySocket.clear();
}

VoronoiSystem* VoronoiSystemComputeController::getSystem(uint64_t socketKey) {
    const auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end() || !it->second || !it->second->isReady()) {
        return nullptr;
    }
    return it->second.get();
}

const VoronoiSystem* VoronoiSystemComputeController::getSystem(uint64_t socketKey) const {
    const auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end() || !it->second || !it->second->isReady()) {
        return nullptr;
    }
    return it->second.get();
}

bool VoronoiSystemComputeController::buildProduct(uint64_t socketKey, VoronoiProduct& outProduct) {
    outProduct = {};
    if (!isConfigured(socketKey)) {
        return false;
    }
    VoronoiSystem* voronoiSystem = getSystem(socketKey);
    if (!voronoiSystem) {
        return false;
    }

    VoronoiSystemBuildStage& buildStage = voronoiSystem->getBuildStage();
    outProduct.globalDisplayRuntimeModelIds = buildStage.getGlobalDisplayRuntimeModelIds();
    outProduct.globalSeedPositions = buildStage.getGlobalSeedPositions();
    outProduct.globalDomainCorners = buildStage.getGlobalDomainCorners();
    outProduct.globalVolumeGridMin = buildStage.getGlobalVolumeGridMin();
    outProduct.globalVolumeGridDim = buildStage.getGlobalVolumeGridDim();
    outProduct.globalVolumeGridCellSize = buildStage.getGlobalVolumeGridCellSize();
    outProduct.fragmentInstanceIds = buildStage.getGlobalFragmentInstanceIds();
    outProduct.fragmentSeedIds = buildStage.getGlobalFragmentSeedIds();
    outProduct.seedToFragment = buildStage.getGlobalSeedToFragment();
    outProduct.fragmentCentroids = buildStage.getGlobalFragmentCentroids();
    outProduct.fragmentVolumes = buildStage.getGlobalFragmentVolumes();
    outProduct.nominalCellSize = buildStage.getGlobalNominalCellSize();
    outProduct.faceFragmentA = buildStage.getGlobalFaceFragmentA();
    outProduct.faceFragmentB = buildStage.getGlobalFaceFragmentB();
    outProduct.faceAreasA = buildStage.getGlobalFaceAreasA();
    outProduct.faceAreaVectorsA = buildStage.getGlobalFaceAreaVectorsA();
    outProduct.faceMomentsA = buildStage.getGlobalFaceMomentsA();
    outProduct.faceAreasB = buildStage.getGlobalFaceAreasB();
    outProduct.faceAreaVectorsB = buildStage.getGlobalFaceAreaVectorsB();
    outProduct.faceMomentsB = buildStage.getGlobalFaceMomentsB();
    outProduct.faceTypes = buildStage.getGlobalFaceTypes();
    outProduct.faceDomainFaces = buildStage.getGlobalFaceDomainFaces();
    outProduct.contactCouplingFaceA = buildStage.getGlobalContactCouplingFaceA();
    outProduct.contactCouplingFaceB = buildStage.getGlobalContactCouplingFaceB();
    outProduct.contactCouplingAreas = buildStage.getGlobalContactCouplingAreas();
    outProduct.boundaryFaceIndices = buildStage.getGlobalBoundaryFaceIndices();
    outProduct.boundaryFaceFragments = buildStage.getGlobalBoundaryFaceFragments();
    outProduct.boundaryFaceDomainFaces = buildStage.getGlobalBoundaryFaceDomainFaces();
    outProduct.instanceFragmentCounts = buildStage.getGlobalInstanceFragmentCounts();
    outProduct.globalFragmentCount = static_cast<uint32_t>(outProduct.fragmentInstanceIds.size());
    outProduct.globalFaceCount = static_cast<uint32_t>(outProduct.faceAreasA.size());
    outProduct.globalContactCouplingCount = static_cast<uint32_t>(outProduct.contactCouplingFaceA.size());
    outProduct.globalBoundaryFaceCount = static_cast<uint32_t>(outProduct.boundaryFaceIndices.size());

    // Validate CPU topology before taking ownership of GPU resources
    if (!outProduct.isValid()) {
        outProduct = {};
        return false;
    }

    // Non-detaching copy of GPU resources, ownership transfers on commit
    buildStage.copyGpu(outProduct);

    HashProduct::seal(outProduct);
    return true;
}


