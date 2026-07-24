#pragma once

#include "UiRuntimeTypes.hpp"

#include <QObject>

class RuntimeNotifier final : public QObject {
    Q_OBJECT

public:
    explicit RuntimeNotifier(QObject* parent = nullptr) : QObject(parent) {}

signals:
    void runtimeReadyChanged(bool ready);
    void viewportStateChanged(const ViewportUiState& state);
    void heatPaletteVisibilityChanged(bool visible);
    void timelineStateChanged(const TimelineUiState& state);
    void simulationStateChanged(const SimulationUiState& state);
    void serialStateChanged(const SerialUiState& state);
    void graphSelectionChanged(NodeGraphNodeId nodeId);
    void inputActionRequested(const InputAction& action);
};
