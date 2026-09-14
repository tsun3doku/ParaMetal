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

void VoronoiSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }
    auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end()) {
        it = systemsBySocket.emplace(
            socketKey,
            std::make_unique<VoronoiSystem>(vulkanDevice, memoryAllocator, commandPool)).first;
    }

    auto& system = it->second;
    if (!system) {
        return;
    }

    const auto configIt = configuredHashes.find(socketKey);
    if (configIt != configuredHashes.end() && configIt->second == config.computeHash) {
        return;
    }
    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return;
    }
    failedComputeHashes.erase(socketKey);

    system->setParams(config.cellSize, config.voxelResolution);
    system->setContactRange(config.contactRange);
    system->clearGeometry();
    system->setSeedPositions(config.pointPositions, config.pointDomainCorners);
    system->setGlobalGeometry(
        config.globalRemeshRuntimeModelIds,
        config.globalRemeshPositions,
        config.globalRemeshTriangleIndices,
        config.globalRemeshSurfacePositions,
        config.globalRemeshSurfaceTriangleIndices,
        config.sdfPadding);
    const bool configured = system->ensureConfigured();

    if (configured) {
        configuredHashes[socketKey] = config.computeHash;
        failedComputeHashes.erase(socketKey);
    } else {
        configuredHashes.erase(socketKey);
        failedComputeHashes[socketKey] = config.computeHash;
    }
}

void VoronoiSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configuredHashes.erase(socketKey);
    failedComputeHashes.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
        systemsBySocket.erase(it);
    }
}

void VoronoiSystemComputeController::disableAll() {
    configuredHashes.clear();
    failedComputeHashes.clear();
    if (!systemsBySocket.empty()) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
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
    outProduct.globalSdfGridMin = buildStage.getGlobalSdfGridMin();
    outProduct.globalSdfGridDim = buildStage.getGlobalSdfGridDim();
    outProduct.globalSdfCellSize = buildStage.getGlobalSdfCellSize();
    outProduct.globalSdfValues = buildStage.getGlobalSdfValues();
    outProduct.globalSdfRuntimeModelIds = buildStage.getGlobalSdfRuntimeModelIds();
    outProduct.fragmentInstanceIds = buildStage.getGlobalFragmentInstanceIds();
    outProduct.fragmentSeedIds = buildStage.getGlobalFragmentSeedIds();
    outProduct.fragmentCentroids = buildStage.getGlobalFragmentCentroids();
    outProduct.fragmentVolumes = buildStage.getGlobalFragmentVolumes();
    outProduct.nominalCellSize = buildStage.getGlobalNominalCellSize();
    outProduct.faceFragmentA = buildStage.getGlobalFaceFragmentA();
    outProduct.faceFragmentB = buildStage.getGlobalFaceFragmentB();
    outProduct.faceAreas = buildStage.getGlobalFaceAreas();
    outProduct.faceOrthogonalDistancesA = buildStage.getGlobalFaceOrthogonalDistancesA();
    outProduct.faceOrthogonalDistancesB = buildStage.getGlobalFaceOrthogonalDistancesB();
    outProduct.faceNormals = buildStage.getGlobalFaceNormals();
    outProduct.contactFaceFragmentA = buildStage.getGlobalContactFaceFragmentA();
    outProduct.contactFaceFragmentB = buildStage.getGlobalContactFaceFragmentB();
    outProduct.contactFaceAreas = buildStage.getGlobalContactFaceAreas();
    outProduct.contactFaceOrthogonalDistancesA = buildStage.getGlobalContactFaceOrthogonalDistancesA();
    outProduct.contactFaceOrthogonalDistancesB = buildStage.getGlobalContactFaceOrthogonalDistancesB();
    outProduct.contactFaceGaps = buildStage.getGlobalContactFaceGaps();
    outProduct.boundaryFaceFragments = buildStage.getGlobalBoundaryFaceFragments();
    outProduct.boundaryFaceDomainFaces = buildStage.getGlobalBoundaryFaceDomainFaces();
    outProduct.boundaryFaceAreas = buildStage.getGlobalBoundaryFaceAreas();
    outProduct.boundaryFaceOrthogonalDistances = buildStage.getGlobalBoundaryFaceOrthogonalDistances();
    outProduct.instanceFragmentCounts = buildStage.getGlobalInstanceFragmentCounts();
    outProduct.globalFragmentCount = static_cast<uint32_t>(outProduct.fragmentInstanceIds.size());
    outProduct.globalFaceCount = static_cast<uint32_t>(outProduct.faceAreas.size());
    outProduct.globalContactFaceCount = static_cast<uint32_t>(outProduct.contactFaceAreas.size());
    outProduct.globalBoundaryFaceCount = static_cast<uint32_t>(outProduct.boundaryFaceAreas.size());

    // Validate CPU topology before taking ownership of GPU resources
    if (!outProduct.isValid()) {
        outProduct = {};
        return false;
    }

    // Safely transfer GPU resources to outProduct
    buildStage.releaseGpuResources(outProduct);

    HashProduct::seal(outProduct);
    return true;
}
