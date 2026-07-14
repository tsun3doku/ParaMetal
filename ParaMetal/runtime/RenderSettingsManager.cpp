#include "RenderSettingsManager.hpp"

void RenderSettingsManager::setWireframeMode(app::WireframeMode mode) {
    std::lock_guard<std::mutex> lock(mutex);
    settings.wireframeMode = mode;
}

void RenderSettingsManager::toggleWireframeMode() {
    std::lock_guard<std::mutex> lock(mutex);
    settings.wireframeMode =
        static_cast<app::WireframeMode>((static_cast<int>(settings.wireframeMode) + 1) % 3);
}

void RenderSettingsManager::toggleTimingOverlay() {
    std::lock_guard<std::mutex> lock(mutex);
    settings.gpuTimingOverlayEnabled = !settings.gpuTimingOverlayEnabled;
}

void RenderSettingsManager::toggleGrid() {
    std::lock_guard<std::mutex> lock(mutex);
    settings.gridEnabled = !settings.gridEnabled;
}

void RenderSettingsManager::onWireframeToggleRequested() {
    toggleWireframeMode();
}

void RenderSettingsManager::onTimingOverlayToggleRequested() {
    toggleTimingOverlay();
}

void RenderSettingsManager::onGridToggleRequested() {
    toggleGrid();
}

app::RenderSettings RenderSettingsManager::getSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return settings;
}
