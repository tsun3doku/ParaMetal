#include "CFDSystemComputeController.hpp"
#include "hash/HashProduct.hpp"
#include "runtime/RuntimeProducts.hpp"
#include <glm/glm.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

namespace cfd {

bool CFDSystemComputeController::canUpdateInPlace(const Config& previous, const Config& next) {
    if (previous.geometryHash == 0 || previous.geometryHash != next.geometryHash) {
        return false;
    }
    if (previous.targetCourant != next.targetCourant) {
        return false;
    }
    // History frame index is duration / time step; changing either reallocates it.
    if (previous.fixedTimeStep != next.fixedTimeStep) {
        return false;
    }
    if (previous.simulationDuration != next.simulationDuration) {
        return false;
    }
    if (glm::length(previous.initialVelocity - next.initialVelocity) > 1e-6f) {
        return false;
    }
    if (std::abs(previous.kinematicViscosity - next.kinematicViscosity) > 1e-12f) {
        return false;
    }
    for (uint32_t i = 0; i < 6; ++i) {
        if (previous.domainBoundaryTypes[i] != next.domainBoundaryTypes[i]) {
            return false;
        }
        if (glm::length(previous.domainBoundaryVelocities[i] - next.domainBoundaryVelocities[i]) > 1e-6f) {
            return false;
        }
    }
    return true;
}

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

    const auto configIt = configuredConfigs.find(socketKey);

    if (configIt != configuredConfigs.end() &&
        configIt->second.computeHash == config.computeHash) {
        cancelCandidate(socketKey);
        return true;
    }

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    auto it = generationSlots.find(socketKey);
    const bool hasSystem = it != generationSlots.end() && it->second.current;

    if (hasSystem && configIt != configuredConfigs.end() &&
        canUpdateInPlace(configIt->second, config) &&
        it->second.current->isInitialized()) {
            pendingConfigs[socketKey] = config;
        pendingConfigs[socketKey].domain = nullptr;
        return true;
    }

    if (!config.domain) {
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    if (it != generationSlots.end() &&
        (it->second.retiring || it->second.candidate)) {
        return false;
    }

    auto candidate = std::make_unique<CFDSystem>(
        vulkanDevice,
        memoryAllocator,
        commandPool,
        maxFramesInFlight);
    if (!candidate->configure(
            *config.domain,
            config.kinematicViscosity,
            config.initialVelocity,
            config.domainBoundaryTypes,
            config.domainBoundaryVelocities,
            config.fixedTimeStep,
            config.targetCourant,
            config.pressureIterations,
            config.pressureTolerance,
            config.simulationDuration)) {
        generationSlots[socketKey].retiring = std::move(candidate);
        failedComputeHashes[socketKey] = config.computeHash;
        std::cerr << "[CFDSystemComputeController] Failed to configure CFDSystem for socket "
                  << socketKey << std::endl;
        return false;
    }

    GenerationSlot& slot = generationSlots[socketKey];
    configuredConfigs.try_emplace(socketKey);
    pendingConfigs[socketKey] = config;
    pendingConfigs[socketKey].domain = nullptr;
    slot.candidate = std::move(candidate);
    return true;
}

void CFDSystemComputeController::remove(uint64_t socketKey) {
    failedComputeHashes.erase(socketKey);
    retireCurrent(socketKey);
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
        if (configuredConfigs.find(it->first) == configuredConfigs.end()) {
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

void CFDSystemComputeController::commitPending(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return;
    }
    GenerationSlot& slot = it->second;
    assert(slot.retiring == nullptr);
    if (slot.retiring != nullptr) {
        return;
    }
    const bool replaced = slot.candidate != nullptr;
    if (replaced) {
        slot.retiring = std::move(slot.current);
        slot.current = std::move(slot.candidate);
    }

    const auto pending = pendingConfigs.find(socketKey);
    if (pending != pendingConfigs.end()) {
        if (!replaced && slot.current) {
                    slot.current->updateParameters(
                pending->second.kinematicViscosity,
                pending->second.initialVelocity,
                pending->second.domainBoundaryTypes,
                pending->second.domainBoundaryVelocities,
                pending->second.fixedTimeStep,
                pending->second.targetCourant);
            slot.current->setPressureSolverParameters(
                pending->second.pressureIterations,
                pending->second.pressureTolerance);
        }
        const auto committed = configuredConfigs.find(socketKey);
        if (committed != configuredConfigs.end()) {
            committed->second = std::move(pending->second);
        }
        pendingConfigs.erase(pending);
    }
    failedComputeHashes.erase(socketKey);
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
    const auto pending = pendingConfigs.find(socketKey);
    if (pending != pendingConfigs.end()) {
        pendingConfigs.erase(pending);
        const auto committed = configuredConfigs.find(socketKey);
        if (committed != configuredConfigs.end() &&
            committed->second.computeHash == 0 &&
            !hasCurrent) {
            configuredConfigs.erase(committed);
        }
    }
}

void CFDSystemComputeController::retireCurrent(uint64_t socketKey) {
    configuredConfigs.erase(socketKey);
    pendingConfigs.erase(socketKey);

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

void CFDSystemComputeController::quiescedClear() noexcept {
    generationSlots.clear();
    configuredConfigs.clear();
    pendingConfigs.clear();
    failedComputeHashes.clear();
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

    outProduct.streamlineVertexBuffer = systemPtr->streamlineVertexBuffer();
    outProduct.streamlineVertexBufferOffset = systemPtr->streamlineVertexBufferOffset();
    outProduct.streamlineVertexCount = systemPtr->streamlineVertexCount();

    outProduct.faceFluxBuffer = runtime.faceFluxBuffer();
    outProduct.faceFluxBufferSize = runtime.getFaceFluxExternalBuffer().getBufferSize();
    outProduct.faceFluxBufferOffset = 0;
    outProduct.faceFluxCount = runtime.getGlobalFaceCount();

    outProduct.boundaryFluxBuffer = runtime.boundaryFluxBuffer();
    outProduct.boundaryFluxBufferSize = runtime.getBoundaryFluxExternalBuffer().getBufferSize();
    outProduct.boundaryFluxBufferOffset = 0;
    outProduct.boundaryFluxCount = runtime.getGlobalBoundaryFaceCount();

    const Config* config = getBuildConfig(socketKey);
    HashProduct::seal(
        outProduct,
        config ? config->geometryHash : 0,
        config ? config->computeHash : 0);

    return outProduct.isValid();
}

const CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) const {
    if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
        return nullptr;
    }
    const auto it = generationSlots.find(socketKey);
    return it != generationSlots.end() ? it->second.current.get() : nullptr;
}

CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) {
    if (configuredConfigs.find(socketKey) == configuredConfigs.end()) {
        return nullptr;
    }
    const auto it = generationSlots.find(socketKey);
    return it != generationSlots.end() ? it->second.current.get() : nullptr;
}

const CFDSystemComputeController::Config* CFDSystemComputeController::getConfig(uint64_t socketKey) const {
    const auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}

const CFDSystemComputeController::Config* CFDSystemComputeController::getBuildConfig(uint64_t socketKey) const {
    const auto pendingIt = pendingConfigs.find(socketKey);
    if (pendingIt != pendingConfigs.end()) {
        return &pendingIt->second;
    }
    const auto configIt = configuredConfigs.find(socketKey);
    return configIt != configuredConfigs.end() ? &configIt->second : nullptr;
}

} // namespace cfd
