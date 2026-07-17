#include "RenderSettingsController.hpp"

#include "RenderSettingsManager.hpp"

RenderSettingsController::RenderSettingsController(RenderSettingsManager* settingsManager,
                                                   QObject* parent)
    : QObject(parent), settingsManager(settingsManager) {
}

void RenderSettingsController::bind(RenderSettingsManager* settingsManager) {
    if (this->settingsManager == settingsManager) {
        return;
    }
    this->settingsManager = settingsManager;
    emit settingsChanged();
}

void RenderSettingsController::setWireframeMode(WireframeMode mode) {
    if (settingsManager) {
        if (settingsManager->getSnapshot().wireframeMode == mode) {
            return;
        }
        settingsManager->setWireframeMode(mode);
        emit settingsChanged();
    }
}

void RenderSettingsController::toggleGrid() {
    if (settingsManager) {
        settingsManager->toggleGrid();
        emit settingsChanged();
    }
}

void RenderSettingsController::onWireframeToggleRequested() {
    if (settingsManager) {
        settingsManager->toggleWireframeMode();
        emit settingsChanged();
    }
}

void RenderSettingsController::onTimingOverlayToggleRequested() {
    if (settingsManager) {
        settingsManager->toggleTimingOverlay();
        emit settingsChanged();
    }
}

void RenderSettingsController::onGridToggleRequested() {
    toggleGrid();
}

app::RenderSettings RenderSettingsController::getSnapshot() const {
    if (settingsManager) {
        return settingsManager->getSnapshot();
    }
    return {};
}
