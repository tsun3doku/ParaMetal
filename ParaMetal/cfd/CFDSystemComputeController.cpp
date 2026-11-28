#include "CFDSystemComputeController.hpp"
#include "hash/HashProduct.hpp"
#include "runtime/RuntimeProducts.hpp"
#include <glm/glm.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

namespace cfd {

CFDSystemComputeController::CFDSystemComputeController(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    CommandPool& cmdPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      commandPool(cmdPool),
      maxFramesInFlight(maxFramesInFlight) {
}

CFDSystemComputeController::~CFDSystemComputeController() {
    generationSlots.clear();
}

bool CFDSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    const auto configIt = configuredHashes.find(socketKey);
    if (configIt != configuredHashes.end() &&
        configIt->second.first == config.geometryHash &&
        configIt->second.second == config.computeHash) {
        cancelCandidate(socketKey);
        return true;
    }

    const auto failedIt = failedHashes.find(socketKey);
    if (failedIt != failedHashes.end() &&
        failedIt->second.first == config.geometryHash &&
        failedIt->second.second == config.computeHash) {
        return false;
    }
    failedHashes.erase(socketKey);

    auto it = generationSlots.find(socketKey);

    if (it != generationSlots.end() &&
        (it->second.retiring || it->second.candidate)) {
        return false;
    }

    if (!config.hasUniqueSeedToSpaceMapping) {
        failedHashes[socketKey] = {config.geometryHash, config.computeHash};
        std::cerr << "[CFDSystemComputeController] CFD requires a unique seed-to-Space mapping for socket "
                  << socketKey << std::endl;
        return false;
    }

    auto candidate = std::make_unique<CFDSystem>(
        vulkanDevice,
        memoryAllocator,
        commandPool,
        maxFramesInFlight);
    if (!candidate->configure(
            config.fragmentInstanceIds,
            config.fragmentSeedIds,
            config.fragmentCentroids,
            config.fragmentVolumes,
            config.faceFragmentA,
            config.faceFragmentB,
            config.faceAreasA,
            config.faceAreasB,
            config.faceAreaVectorsA,
            config.faceAreaVectorsB,
            config.faceMomentsA,
            config.faceMomentsB,
            config.faceTypes,
            config.faceDomainFaces,
            config.boundaryFaceIndices,
            config.faceLoopOffsets,
            config.loopVertexOffsets,
            config.loopVertices,
            config.seedToSpaceIndex,
            config.spaceToFragmentIndex,
            config.hasUniqueSeedToSpaceMapping,
            config.seedIdGridBuffer,
            config.seedIdGridBufferOffset,
            config.kinematicViscosity,
            config.initialVelocity,
            config.domainBoundaryTypes,
            config.domainBoundaryVelocities,
            config.fixedTimeStep,
            config.targetCourant,
            config.pressureIterations,
            config.pressureTolerance,
            config.simulationDuration,
            config.domainMin,
            config.domainMax,
            config.volumeGridMin,
            config.volumeGridDim,
            config.volumeGridCellSize)) {
        generationSlots[socketKey].retiring = std::move(candidate);
        failedHashes[socketKey] = {config.geometryHash, config.computeHash};
        std::cerr << "[CFDSystemComputeController] Failed to configure CFDSystem for socket "
                  << socketKey << std::endl;
        return false;
    }

    GenerationSlot& slot = generationSlots[socketKey];
    configuredHashes.try_emplace(socketKey, 0u, 0u);
    pendingHashes[socketKey] = {config.geometryHash, config.computeHash};
    slot.candidate = std::move(candidate);
    return true;
}

bool CFDSystemComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return true;
    }
    failedHashes.erase(socketKey);
    return retireCurrent(socketKey);
}

bool CFDSystemComputeController::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : generationSlots) {
        (void)socketKey;
        if (slot.retiring) return true;
    }
    return false;
}

void CFDSystemComputeController::clearAllRetiring() {
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

bool CFDSystemComputeController::commitPending(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return false;
    }
    GenerationSlot& slot = it->second;
    if (slot.retiring != nullptr) {
        return false;
    }
    if (slot.candidate == nullptr) {
        return false;
    }
    slot.retiring = std::move(slot.current);
    slot.current = std::move(slot.candidate);

    const auto pending = pendingHashes.find(socketKey);
    if (pending != pendingHashes.end()) {
        const auto committed = configuredHashes.find(socketKey);
        if (committed != configuredHashes.end()) {
            committed->second = std::move(pending->second);
        }
        pendingHashes.erase(pending);
    }
    failedHashes.erase(socketKey);
    return true;
}

void CFDSystemComputeController::cancelCandidate(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it != generationSlots.end() && it->second.candidate) {
        assert(it->second.retiring == nullptr);
        if (!it->second.retiring) {
            it->second.retiring = std::move(it->second.candidate);
        }
    }
    const bool hasCurrent = it != generationSlots.end() && it->second.current;
    const auto pending = pendingHashes.find(socketKey);
    if (pending != pendingHashes.end()) {
        pendingHashes.erase(pending);
        const auto committed = configuredHashes.find(socketKey);
        if (committed != configuredHashes.end() &&
            committed->second.second == 0 &&
            !hasCurrent) {
            configuredHashes.erase(committed);
        }
    }
}

bool CFDSystemComputeController::retireCurrent(uint64_t socketKey) {
    configuredHashes.erase(socketKey);
    pendingHashes.erase(socketKey);

    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return true;
    }
    GenerationSlot& slot = it->second;
    if (slot.retiring) {
        return false;
    }
    if (slot.candidate) {
        slot.retiring = std::move(slot.candidate);
        return !slot.current;
    }
    if (slot.current) {
        slot.retiring = std::move(slot.current);
    }
    return true;
}

void CFDSystemComputeController::quiescedClear() noexcept {
    generationSlots.clear();
    configuredHashes.clear();
    pendingHashes.clear();
    failedHashes.clear();
}

void CFDSystemComputeController::releaseWaits() {
    for (auto& [key, slot] : generationSlots) {
        (void)key;
        if (slot.current) slot.current->releaseWaits();
        if (slot.retiring) slot.retiring->releaseWaits();
        if (slot.candidate) slot.candidate->releaseWaits();
    }
}

bool CFDSystemComputeController::buildProduct(uint64_t socketKey, CFDProduct& outProduct) {
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return false;
    }
    CFDSystem* systemPtr = it->second.candidate
        ? it->second.candidate.get()
        : it->second.current.get();
    if (!systemPtr || !systemPtr->isInitialized()) {
        return false;
    }

    const auto& runtime = systemPtr->getRuntime();
    outProduct.fluidNodeCount = runtime.getFluidNodeCount();
    outProduct.globalFragmentToFluidNode = systemPtr->getGlobalFragmentToFluidNode();

    outProduct.streamlineVertexBuffer = systemPtr->streamlineVertexBuffer();
    outProduct.streamlineVertexBufferOffset = systemPtr->streamlineVertexBufferOffset();
    outProduct.streamlineVertexCount = systemPtr->streamlineVertexCount();

    outProduct.velocityBuffer = systemPtr->getDisplayVelocityBuffer();
    outProduct.velocityBufferOffset = systemPtr->getDisplayVelocityBufferOffset();
    outProduct.gradientBuffer = systemPtr->getDisplayGradientBuffer();
    outProduct.gradientBufferOffset = systemPtr->getDisplayGradientBufferOffset();
    outProduct.centroidBuffer = systemPtr->getFluidCentroidBuffer();
    outProduct.centroidBufferOffset = systemPtr->getFluidCentroidBufferOffset();

    outProduct.faceFluxBuffer = runtime.faceFluxBuffer();
    outProduct.faceFluxBufferSize = runtime.getFaceFluxExternalBuffer().getBufferSize();
    outProduct.faceFluxBufferOffset = 0;
    outProduct.faceFluxCount = runtime.getGlobalFaceCount();

    outProduct.boundaryFluxBuffer = runtime.boundaryFluxBuffer();
    outProduct.boundaryFluxBufferSize = runtime.getBoundaryFluxExternalBuffer().getBufferSize();
    outProduct.boundaryFluxBufferOffset = 0;
    outProduct.boundaryFluxCount = runtime.getGlobalBoundaryFaceCount();

    uint64_t geometryHash = 0;
    uint64_t simulationHash = 0;
    if (!getBuildHashes(socketKey, geometryHash, simulationHash)) {
        return false;
    }
    HashProduct::seal(outProduct, geometryHash, simulationHash);

    return outProduct.isValid();
}

const CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) const {
    if (configuredHashes.find(socketKey) == configuredHashes.end()) {
        return nullptr;
    }
    const auto it = generationSlots.find(socketKey);
    return it != generationSlots.end() ? it->second.current.get() : nullptr;
}

CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) {
    if (configuredHashes.find(socketKey) == configuredHashes.end()) {
        return nullptr;
    }
    const auto it = generationSlots.find(socketKey);
    return it != generationSlots.end() ? it->second.current.get() : nullptr;
}

bool CFDSystemComputeController::getBuildHashes(
    uint64_t socketKey,
    uint64_t& geometryHash,
    uint64_t& simulationHash) const {
    const auto pendingIt = pendingHashes.find(socketKey);
    if (pendingIt != pendingHashes.end()) {
        geometryHash = pendingIt->second.first;
        simulationHash = pendingIt->second.second;
        return true;
    }
    const auto configuredIt = configuredHashes.find(socketKey);
    if (configuredIt == configuredHashes.end()) {
        return false;
    }
    geometryHash = configuredIt->second.first;
    simulationHash = configuredIt->second.second;
    return true;
}

} // namespace cfd


