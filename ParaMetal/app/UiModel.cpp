#include "UiModel.hpp"

#include <algorithm>
#include <cmath>

void ViewportUiModel::setWireframeMode(int mode) {
    Q_ASSERT(mode >= static_cast<int>(app::WireframeMode::Off));
    Q_ASSERT(mode <= static_cast<int>(app::WireframeMode::Shaded));
    if (mode < static_cast<int>(app::WireframeMode::Off) ||
        mode > static_cast<int>(app::WireframeMode::Shaded)) return;
    emit wireframeModeRequested(static_cast<app::WireframeMode>(mode));
}

void ViewportUiModel::setGridEnabled(bool enabled) {
    emit gridEnabledRequested(enabled);
}

void ViewportUiModel::applyState(const ViewportUiState& updated) {
    if (state == updated) return;
    state = updated;
    emit stateChanged();
}

void HeatPaletteUiModel::setPaletteId(int value) {
    value = std::clamp(value, 0, 3);
    if (state.paletteId == value) return;
    state.paletteId = value;
    emit stateChanged();
    emit paletteRequested(value);
}

void HeatPaletteUiModel::setUnits(int value) {
    value = std::clamp(value, 0, 2);
    if (state.units == value) return;
    state.units = value;
    emit stateChanged();
}

void HeatPaletteUiModel::setRange(double minimum, double maximum) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || maximum <= minimum) return;
    state.minimumC = minimum;
    state.maximumC = maximum;
    emit stateChanged();
    emit rangeRequested(static_cast<float>(minimum), static_cast<float>(maximum));
}

void HeatPaletteUiModel::setPosition(double newX, double newY) {
    if (state.x == newX && state.y == newY) return;
    state.x = newX;
    state.y = newY;
    emit stateChanged();
}

void HeatPaletteUiModel::applyVisibility(bool value) {
    if (state.visible == value) return;
    state.visible = value;
    emit stateChanged();
}

void TimelineUiModel::setPlaying(bool value) {
    emit playingRequested(value);
}
void TimelineUiModel::reset() { emit resetRequested(); }
void TimelineUiModel::scrub(int frame) { emit scrubRequested(static_cast<uint32_t>(std::max(0, frame))); }
void TimelineUiModel::step(int delta) { emit stepRequested(delta); }

void TimelineUiModel::applyState(const TimelineUiState& updated) {
    if (state == updated) return;
    state = updated;
    emit stateChanged();
}

void RuntimeStatusUiModel::setReady(bool value) {
    if (runtimeReady == value) return;
    runtimeReady = value;
    emit readyChanged();
}

void RuntimeStatusUiModel::applySimulation(const SimulationUiState& updated) {
    if (simulation == updated) return;
    simulation = updated;
    emit simulationChanged();
}

void RuntimeStatusUiModel::applySerial(const SerialUiState& updated) {
    if (serial == updated) return;
    serial = updated;
    emit serialChanged();
}

ConsoleModel::ConsoleModel(QObject* parent) : QObject(parent) {
    terminalOutput = QStringLiteral("ParaMetal Console\nInitializing Python...\n\n");
}

void ConsoleModel::setPythonVersion(const QString& version) {
    terminalOutput = QStringLiteral("ParaMetal Console\nPython %1\nparametal imported as pm\n\nHelp: api(), registry()\n\n").arg(version);
    emit outputChanged();
}

void ConsoleModel::execute(const QString& source) {
    if (requestPending) return;
    if (waitingForMoreInput) {
        pythonBuffer += QStringLiteral("\n") + source;
        terminalOutput += QStringLiteral("... ") + source + QStringLiteral("\n");
    } else {
        pythonBuffer = source;
        terminalOutput += QStringLiteral(">>> ") + source + QStringLiteral("\n");
    }
    requestPending = true;
    emit outputChanged();
    emit executeRequested(pythonBuffer);
}

void ConsoleModel::clear() {
    terminalOutput.clear();
    emit outputChanged();
}

void ConsoleModel::applyResult(const PythonResult& result) {
    requestPending = false;
    waitingForMoreInput = result.incomplete;
    terminalOutput += result.output;
    terminalOutput += result.error;
    if (!waitingForMoreInput) pythonBuffer.clear();
    emit outputChanged();
}

UiModel::UiModel(QObject* parent)
    : QObject(parent),
      viewportModel(nullptr), heatPaletteModel(nullptr), timelineModel(nullptr), runtimeModel(nullptr), graphModel(nullptr),
      consoleModel(nullptr) {
}
