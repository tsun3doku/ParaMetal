#include "VoronoiSystemComputeController.hpp"

#include "voronoi/VoronoiSystem.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "hash/HashProduct.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <iostream>

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
        std::cerr << "[VoronoiSystemComputeController] generation slot occupied for socket "
                  << socketKey << std::endl;
        return false;
    }

    auto candidate = std::make_unique<VoronoiSystem>(vulkanDevice, memoryAllocator, commandPool);
    candidate->setParams(config.cellSize, config.voxelResolution);
    candidate->setLloydParams(
        config.lloydIterations,
        config.lloydVolumeThreshold,
        config.lloydDamping);
    candidate->setContactRange(config.contactRange);
    candidate->setSeedPositions(config.pointPositions, config.pointDomainCorners);
    candidate->setGlobalGeometry(
        config.globalRemeshRuntimeModelIds,
        config.globalRemeshPositions,
        config.globalRemeshTriangleIndices,
        config.globalRemeshSurfacePositions,
        config.globalRemeshSurfaceTriangleIndices,
        config.globalRemeshWorldToModel,
        config.volumePadding);

    if (!candidate->ensureConfigured()) {
        generationSlots[socketKey].retiring = std::move(candidate);
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    GenerationSlot& slot = generationSlots[socketKey];
    configuredHashes.try_emplace(socketKey, 0);
    pendingHashes[socketKey] = config.computeHash;
    slot.candidate = std::move(candidate);
    failedComputeHashes.erase(socketKey);
    return true;
}

void VoronoiSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    failedComputeHashes.erase(socketKey);
    retireCurrent(socketKey);
}

bool VoronoiSystemComputeController::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : generationSlots) {
        (void)socketKey;
        if (slot.retiring) return true;
    }
    return false;
}

void VoronoiSystemComputeController::clearAllRetiring() {
    for (auto it = generationSlots.begin(); it != generationSlots.end();) {
        GenerationSlot& slot = it->second;
        slot.retiring.reset();
        if (configuredHashes.find(it->first) == configuredHashes.end()) {
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

void VoronoiSystemComputeController::commitPending(uint64_t socketKey) noexcept {
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

    const auto pending = pendingHashes.find(socketKey);
    if (pending != pendingHashes.end()) {
        const auto committed = configuredHashes.find(socketKey);
        if (committed != configuredHashes.end()) {
            committed->second = pending->second;
        }
        pendingHashes.erase(pending);
    }
    failedComputeHashes.erase(socketKey);
}

void VoronoiSystemComputeController::cancelCandidate(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it != generationSlots.end() && it->second.candidate) {
        assert(it->second.retiring == nullptr);
        if (!it->second.retiring) {
            it->second.retiring = std::move(it->second.candidate);
        }
        if (!it->second.current) {
            const auto committed = configuredHashes.find(socketKey);
            if (committed != configuredHashes.end() && committed->second == 0) {
                configuredHashes.erase(committed);
            }
        }
    }
    pendingHashes.erase(socketKey);
}

void VoronoiSystemComputeController::retireCurrent(uint64_t socketKey) {
    configuredHashes.erase(socketKey);
    pendingHashes.erase(socketKey);

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

void VoronoiSystemComputeController::quiescedClear() noexcept {
    generationSlots.clear();
    configuredHashes.clear();
    pendingHashes.clear();
    failedComputeHashes.clear();
}

VoronoiSystem* VoronoiSystemComputeController::getSystem(uint64_t socketKey) {
    if (configuredHashes.find(socketKey) == configuredHashes.end()) {
        return nullptr;
    }
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end() || !it->second.current || !it->second.current->isReady()) {
        return nullptr;
    }
    return it->second.current.get();
}

bool VoronoiSystemComputeController::buildProduct(uint64_t socketKey, VoronoiProduct& outProduct) {
    outProduct = {};
    const auto slotIt = generationSlots.find(socketKey);
    if (slotIt == generationSlots.end()) {
        return false;
    }
    VoronoiSystem* voronoiSystem = slotIt->second.candidate
        ? slotIt->second.candidate.get()
        : slotIt->second.current.get();
    if (!voronoiSystem || !voronoiSystem->isReady()) {
        return false;
    }

    VoronoiSystemBuildStage& buildStage = voronoiSystem->getBuildStage();
    outProduct.globalSeedPositions = buildStage.getGlobalSeedPositions();
    outProduct.globalDomainCorners = buildStage.getGlobalDomainCorners();
    outProduct.globalVolumeGridMin = buildStage.getGlobalVolumeGridMin();
    outProduct.globalVolumeGridDim = buildStage.getGlobalVolumeGridDim();
    outProduct.globalVolumeGridCellSize = buildStage.getGlobalVolumeGridCellSize();
    outProduct.fragmentInstanceIds = buildStage.getGlobalFragmentInstanceIds();
    outProduct.fragmentSeedIds = buildStage.getGlobalFragmentSeedIds();
    outProduct.cellFragmentOffsets = buildStage.getGlobalCellFragmentOffsets();
    outProduct.cellFragments = buildStage.getGlobalCellFragments();
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
    outProduct.contactCouplingFragmentA = buildStage.getGlobalContactCouplingFragmentA();
    outProduct.contactCouplingFaceB = buildStage.getGlobalContactCouplingFaceB();
    outProduct.contactCouplingFragmentB = buildStage.getGlobalContactCouplingFragmentB();
    outProduct.contactCouplingAreas = buildStage.getGlobalContactCouplingAreas();
    outProduct.boundaryFaceIndices = buildStage.getGlobalBoundaryFaceIndices();
    outProduct.boundaryFaceFragments = buildStage.getGlobalBoundaryFaceFragments();
    outProduct.boundaryFaceDomainFaces = buildStage.getGlobalBoundaryFaceDomainFaces();
    outProduct.instanceFragmentCounts = buildStage.getGlobalInstanceFragmentCounts();
    outProduct.globalFragmentCount = static_cast<uint32_t>(outProduct.fragmentInstanceIds.size());
    outProduct.globalFaceCount = static_cast<uint32_t>(outProduct.faceAreasA.size());
    outProduct.globalContactCouplingCount = static_cast<uint32_t>(outProduct.contactCouplingFaceA.size());
    outProduct.globalBoundaryFaceCount = static_cast<uint32_t>(outProduct.boundaryFaceIndices.size());

    if (!outProduct.isValid()) {
        outProduct = {};
        return false;
    }

    buildStage.copyGpu(outProduct);

    HashProduct::seal(outProduct);
    return true;
}
