#include "HeatDisplayController.hpp"

#include "render/HeatOverlayRenderer.hpp"
#include "renderers/HeatVolumeRenderer.hpp"

#include <iostream>

void HeatDisplayController::setOverlayRenderer(render::HeatOverlayRenderer* updatedOverlayRenderer) {
    overlayRenderer = updatedOverlayRenderer;
}

void HeatDisplayController::setVolumeRenderer(render::HeatVolumeRenderer* updatedVolumeRenderer) {
    volumeRenderer = updatedVolumeRenderer;
}

void HeatDisplayController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return;
    }

    syncedSockets.insert(socketKey);

    const bool previewEnabled =
        config.authoredActive &&
        config.anyVisible() &&
        config.isValid();
    if (!previewEnabled) {
        remove(socketKey);
        return;
    }

    const auto existingIt = configsBySocket.find(socketKey);
    if (existingIt != configsBySocket.end() && existingIt->second.displayHash == config.displayHash) {
        return;
    }

    if (overlayRenderer) {
        overlayRenderer->apply(socketKey, config);
    }

    if (volumeRenderer) {
        if (config.showFluidVolume && config.hasFluidVolume && config.fluidTemperatureVolumeView != VK_NULL_HANDLE) {
            volumeRenderer->setEnabled(true);
            volumeRenderer->setOpacity(config.fluidVolumeOpacity);
            if (overlayRenderer) {
                volumeRenderer->setPalette(overlayRenderer->getPalette());
                volumeRenderer->setRange(overlayRenderer->getMinTemperature(), overlayRenderer->getMaxTemperature());
            }
            if (config.hasGlobalDomain) {
                volumeRenderer->setVolumeDomain(config.globalDomainCorners);
            }
            volumeRenderer->setVolumeTexture(
                config.fluidTemperatureVolumeView,
                config.fluidTemperatureVolumeSampler,
                config.globalVolumeGridMin,
                config.globalVolumeGridDim,
                config.globalVolumeGridCellSize);
        } else {
            volumeRenderer->setEnabled(false);
        }
    }

    configsBySocket[socketKey] = config;
}

void HeatDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    if (overlayRenderer) {
        overlayRenderer->remove(socketKey);
    }

    if (volumeRenderer) {
        volumeRenderer->setEnabled(false);
    }

    configsBySocket.erase(socketKey);
}

void HeatDisplayController::finalizeSync() {
    std::vector<uint64_t> staleSocketKeys;
    staleSocketKeys.reserve(configsBySocket.size());
    for (const auto& [socketKey, config] : configsBySocket) {
        (void)config;
        if (syncedSockets.find(socketKey) == syncedSockets.end()) {
            staleSocketKeys.push_back(socketKey);
        }
    }

    for (uint64_t socketKey : staleSocketKeys) {
        remove(socketKey);
    }

    syncedSockets.clear();
}