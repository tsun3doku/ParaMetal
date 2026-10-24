#include "CFDSystemComputeController.hpp"
#include "runtime/RuntimeProducts.hpp"
#include <glm/glm.hpp>
#include <iostream>

namespace cfd {

bool CFDSystemComputeController::canUpdateInPlace(const Config& previous, const Config& next) {
    if (previous.geometryHash == 0 || previous.geometryHash != next.geometryHash) {
        return false;
    }
    if (previous.targetCourant != next.targetCourant) {
        return false;
    }
    if (glm::length(previous.initialVelocity - next.initialVelocity) > 1e-6f) {
        return false;
    }
    for (uint32_t i = 0; i < 6; ++i) {
        if (previous.domainBoundaryTypes[i] != next.domainBoundaryTypes[i]) {
            return false;
        }
    }
    return true;
}

CFDSystemComputeController::CFDSystemComputeController(
    VulkanDevice& device,
    CommandPool& cmdPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(device),
      commandPool(cmdPool),
      maxFramesInFlight(maxFramesInFlight) {
}

CFDSystemComputeController::~CFDSystemComputeController() {
    systemsBySocket.clear();
    retainedSystems.clear();
}

void CFDSystemComputeController::applyRuntimeState(CFDSystem& system, const Config& config) {
    system.setActive(config.active);
    system.setPaused(!timelinePlaying);
}

void CFDSystemComputeController::setTimelinePlaying(bool playing) {
    timelinePlaying = playing;
    for (auto& [key, system] : systemsBySocket) {
        if (system) {
            system->setPaused(!playing);
        }
    }
}

void CFDSystemComputeController::resetTimeline() {
    timelinePlaying = false;
    for (auto& [key, system] : systemsBySocket) {
        if (system) {
            system->setPaused(true);
            system->resetSimulationState();
        }
    }
}

void CFDSystemComputeController::scrubTimeline(uint32_t frame) {
    (void)frame;
    timelinePlaying = false;
    for (auto& [key, system] : systemsBySocket) {
        if (system) {
            system->setPaused(true);
        }
    }
}

bool CFDSystemComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    if (!config.active) {
        remove(socketKey);
        return true;
    }

    const auto configIt = configuredConfigs.find(socketKey);

    // Tier 1: computeHash matches -> simulation state and inputs unchanged
    if (configIt != configuredConfigs.end() && configIt->second.computeHash == config.computeHash) {
        const auto it = systemsBySocket.find(socketKey);
        if (it != systemsBySocket.end() && it->second) {
            applyRuntimeState(*it->second, config);
        }
        return true;
    }

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    auto it = systemsBySocket.find(socketKey);
    const bool hasSystem = it != systemsBySocket.end() && it->second;

    // Tier 2 in-place updates require unchanging boundary types, velocities, and Courant
    if (hasSystem && configIt != configuredConfigs.end() &&
        canUpdateInPlace(configIt->second, config) &&
        it->second->isInitialized()) {
        it->second->updateParameters(
            config.kinematicViscosity,
            config.initialVelocity,
            config.domainBoundaryTypes,
            config.domainBoundaryVelocities,
            config.fixedTimeStep,
            config.targetCourant);
        it->second->setPressureSolverParameters(config.pressureIterations, config.pressureTolerance);

        configuredConfigs[socketKey] = config;
        configuredConfigs[socketKey].domain = nullptr;

        applyRuntimeState(*it->second, config);
        return true;
    }

    // Tier 3 replacement builds candidate off to side while active system remains live
    if (!config.domain) {
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    auto candidate = std::make_unique<CFDSystem>(
        vulkanDevice,
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
            config.pressureTolerance)) {
        // Retain failed candidate to avoid destroying resources with in-flight CUDA work
        retainedSystems.push_back(std::move(candidate));
        failedComputeHashes[socketKey] = config.computeHash;
        std::cerr << "[CFDSystemComputeController] Failed to configure CFDSystem for socket "
                  << socketKey << std::endl;
        return false;
    }

    if (it != systemsBySocket.end()) {
        if (it->second) {
            retainedSystems.push_back(std::move(it->second));
        }
        it->second = std::move(candidate);
    } else {
        systemsBySocket.emplace(socketKey, std::move(candidate));
    }

    configuredConfigs[socketKey] = config;
    configuredConfigs[socketKey].domain = nullptr;

    applyRuntimeState(*systemsBySocket[socketKey], config);
    return true;
}

void CFDSystemComputeController::remove(uint64_t socketKey) {
    configuredConfigs.erase(socketKey);
    failedComputeHashes.erase(socketKey);
    auto it = systemsBySocket.find(socketKey);
    if (it != systemsBySocket.end()) {
        if (it->second) {
            it->second->setPaused(true);
            it->second->setActive(false);
            retainedSystems.push_back(std::move(it->second));
        }
        systemsBySocket.erase(it);
    }
}

void CFDSystemComputeController::disableAll() {
    for (auto& [key, sys] : systemsBySocket) {
        (void)key;
        if (sys) {
            sys->setActive(false);
            sys->setPaused(true);
            retainedSystems.push_back(std::move(sys));
        }
    }
    systemsBySocket.clear();
}

void CFDSystemComputeController::releaseWaits() {
    for (auto& [key, sys] : systemsBySocket) {
        (void)key;
        if (sys) {
            sys->releaseWaits();
        }
    }
    for (auto& sys : retainedSystems) {
        if (sys) {
            sys->releaseWaits();
        }
    }
}

bool CFDSystemComputeController::buildProduct(uint64_t socketKey, CFDProduct& outProduct) {
    auto it = systemsBySocket.find(socketKey);
    if (it == systemsBySocket.end() || !it->second || !it->second->isInitialized()) {
        return false;
    }

    const auto& runtime = it->second->getRuntime();
    outProduct.fluidNodeCount = runtime.getFluidNodeCount();

    outProduct.velocityBuffer = runtime.velocityBuffer();
    outProduct.velocityBufferSize = runtime.getVelocityExternalBuffer().getBufferSize();
    outProduct.velocityBufferOffset = 0;

    outProduct.faceFluxBuffer = runtime.faceFluxBuffer();
    outProduct.faceFluxBufferSize = runtime.getFaceFluxExternalBuffer().getBufferSize();
    outProduct.faceFluxBufferOffset = 0;
    outProduct.faceFluxCount = runtime.getGlobalFaceCount();

    outProduct.boundaryFluxBuffer = runtime.boundaryFluxBuffer();
    outProduct.boundaryFluxBufferSize = runtime.getBoundaryFluxExternalBuffer().getBufferSize();
    outProduct.boundaryFluxBufferOffset = 0;
    outProduct.boundaryFluxCount = runtime.getGlobalBoundaryFaceCount();

    const auto cfgIt = configuredConfigs.find(socketKey);
    if (cfgIt != configuredConfigs.end()) {
        outProduct.hashes.simulation = cfgIt->second.computeHash;
        outProduct.hashes.geometry = cfgIt->second.geometryHash;
        outProduct.hashes.full = cfgIt->second.computeHash;
    }

    return outProduct.isValid();
}

const CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) const {
    auto it = systemsBySocket.find(socketKey);
    return it != systemsBySocket.end() ? it->second.get() : nullptr;
}

CFDSystem* CFDSystemComputeController::getSystem(uint64_t socketKey) {
    auto it = systemsBySocket.find(socketKey);
    return it != systemsBySocket.end() ? it->second.get() : nullptr;
}

const CFDSystemComputeController::Config* CFDSystemComputeController::getConfig(uint64_t socketKey) const {
    auto it = configuredConfigs.find(socketKey);
    return it != configuredConfigs.end() ? &it->second : nullptr;
}

} // namespace cfd
