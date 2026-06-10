#include "PointDisplayController.hpp"

#include "render/PointOverlayRenderer.hpp"

void PointDisplayController::setOverlayRenderer(render::PointOverlayRenderer* updatedOverlayRenderer) {
    overlayRenderer = updatedOverlayRenderer;
}

void PointDisplayController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0 || !config.isValid()) {
        remove(socketKey);
        return;
    }

    configsBySocket[socketKey] = config;
}

void PointDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configsBySocket.erase(socketKey);
}

void PointDisplayController::finalizeSync() {
    if (!overlayRenderer) {
        return;
    }

    std::unordered_set<uint64_t> nextSynced;
    for (const auto& [socketKey, config] : configsBySocket) {
        overlayRenderer->apply(socketKey, config);
        nextSynced.insert(socketKey);
    }

    for (uint64_t socketKey : syncedSockets) {
        if (nextSynced.find(socketKey) == nextSynced.end()) {
            overlayRenderer->remove(socketKey);
        }
    }

    syncedSockets = std::move(nextSynced);
}
