#pragma once

#include "NodeGraphModel.hpp"
#include "UiRuntimeTypes.hpp"

#include <QObject>
#include <QString>

class ViewportUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int wireframeMode READ wireframeMode NOTIFY stateChanged)
    Q_PROPERTY(bool gridEnabled READ gridEnabled NOTIFY stateChanged)
public:
    explicit ViewportUiModel(QObject* parent = nullptr) : QObject(parent) {}
    int wireframeMode() const { return static_cast<int>(state.wireframeMode); }
    bool gridEnabled() const { return state.gridEnabled; }
    Q_INVOKABLE void setWireframeMode(int mode);
    Q_INVOKABLE void setGridEnabled(bool enabled);
public slots:
    void applyState(const ViewportUiState& updated);
signals:
    void stateChanged();
    void wireframeModeRequested(app::WireframeMode mode);
    void gridEnabledRequested(bool enabled);
private:
    ViewportUiState state{};
};

class HeatPaletteUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool visible READ visible NOTIFY stateChanged)
    Q_PROPERTY(int paletteId READ paletteId NOTIFY stateChanged)
    Q_PROPERTY(int units READ units NOTIFY stateChanged)
    Q_PROPERTY(double minimumC READ minimumC NOTIFY stateChanged)
    Q_PROPERTY(double maximumC READ maximumC NOTIFY stateChanged)
    Q_PROPERTY(double x READ x NOTIFY stateChanged)
    Q_PROPERTY(double y READ y NOTIFY stateChanged)
public:
    explicit HeatPaletteUiModel(QObject* parent = nullptr) : QObject(parent) {}
    bool visible() const { return state.visible; }
    int paletteId() const { return state.paletteId; }
    int units() const { return state.units; }
    double minimumC() const { return state.minimumC; }
    double maximumC() const { return state.maximumC; }
    double x() const { return state.x; }
    double y() const { return state.y; }
    Q_INVOKABLE void setPaletteId(int value);
    Q_INVOKABLE void setUnits(int value);
    Q_INVOKABLE void setRange(double minimumC, double maximumC);
    Q_INVOKABLE void setPosition(double x, double y);
public slots:
    void applyVisibility(bool visible);
signals:
    void stateChanged();
    void paletteRequested(int palette);
    void rangeRequested(float minimum, float maximum);
private:
    HeatPaletteUiState state{};
};

class TimelineUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool playing READ playing NOTIFY stateChanged)
    Q_PROPERTY(int currentFrame READ currentFrame NOTIFY stateChanged)
    Q_PROPERTY(int frameCount READ frameCount NOTIFY stateChanged)
    Q_PROPERTY(int recordedFrames READ recordedFrames NOTIFY stateChanged)
    Q_PROPERTY(int startFrame READ startFrame NOTIFY stateChanged)
    Q_PROPERTY(int endFrame READ endFrame NOTIFY stateChanged)
    Q_PROPERTY(double currentSeconds READ currentSeconds NOTIFY stateChanged)
    Q_PROPERTY(double durationSeconds READ durationSeconds NOTIFY stateChanged)
public:
    explicit TimelineUiModel(QObject* parent = nullptr) : QObject(parent) {}
    bool playing() const { return state.playing; }
    int currentFrame() const { return static_cast<int>(state.currentFrame); }
    int frameCount() const { return static_cast<int>(state.frameCount); }
    int recordedFrames() const { return static_cast<int>(state.recordedFrames); }
    int startFrame() const { return static_cast<int>(state.startFrame); }
    int endFrame() const { return static_cast<int>(state.endFrame); }
    double currentSeconds() const { return state.currentSeconds; }
    double durationSeconds() const { return state.durationSeconds; }
    Q_INVOKABLE void setPlaying(bool playing);
    Q_INVOKABLE void reset();
    Q_INVOKABLE void scrub(int frame);
    Q_INVOKABLE void step(int delta);
public slots:
    void applyState(const TimelineUiState& updated);
signals:
    void stateChanged();
    void playingRequested(bool playing);
    void resetRequested();
    void scrubRequested(uint32_t frame);
    void stepRequested(int delta);
private:
    TimelineUiState state{};
};

class RuntimeStatusUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(bool simulationActive READ simulationActive NOTIFY simulationChanged)
    Q_PROPERTY(bool simulationPaused READ simulationPaused NOTIFY simulationChanged)
    Q_PROPERTY(QString serialConnectionText READ serialConnectionText NOTIFY serialChanged)
    Q_PROPERTY(QString serialTemperatureText READ serialTemperatureText NOTIFY serialChanged)
    Q_PROPERTY(QString serialPollingRateText READ serialPollingRateText NOTIFY serialChanged)
public:
    explicit RuntimeStatusUiModel(QObject* parent = nullptr) : QObject(parent) {}
    bool ready() const { return runtimeReady; }
    bool simulationActive() const { return simulation.active; }
    bool simulationPaused() const { return simulation.paused; }
    QString serialConnectionText() const { return serial.connection; }
    QString serialTemperatureText() const { return serial.temperature; }
    QString serialPollingRateText() const { return serial.pollingRate; }
public slots:
    void setReady(bool ready);
    void applySimulation(const SimulationUiState& updated);
    void applySerial(const SerialUiState& updated);
signals:
    void readyChanged();
    void simulationChanged();
    void serialChanged();
private:
    bool runtimeReady = false;
    SimulationUiState simulation{};
    SerialUiState serial{};
};

class ConsoleModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString output READ output NOTIFY outputChanged)
public:
    explicit ConsoleModel(QObject* parent = nullptr);
    QString output() const { return terminalOutput; }
    Q_INVOKABLE void execute(const QString& source);
    Q_INVOKABLE void clear();
    Q_INVOKABLE void resetDefaultGraph() { emit resetGraphRequested(); }
    void setPythonVersion(const QString& version);
public slots:
    void applyResult(const PythonResult& result);
signals:
    void outputChanged();
    void executeRequested(const QString& source);
    void resetGraphRequested();
private:
    QString terminalOutput;
    QString pythonBuffer;
    bool waitingForMoreInput = false;
    bool requestPending = false;
};

class UiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(ViewportUiModel* viewport READ viewport CONSTANT)
    Q_PROPERTY(HeatPaletteUiModel* heatPalette READ heatPalette CONSTANT)
    Q_PROPERTY(TimelineUiModel* timeline READ timeline CONSTANT)
    Q_PROPERTY(RuntimeStatusUiModel* runtime READ runtime CONSTANT)
    Q_PROPERTY(NodeGraphModel* nodeGraph READ nodeGraph CONSTANT)
    Q_PROPERTY(ConsoleModel* console READ console CONSTANT)
public:
    explicit UiModel(QObject* parent = nullptr);
    ViewportUiModel* viewport() { return &viewportModel; }
    HeatPaletteUiModel* heatPalette() { return &heatPaletteModel; }
    TimelineUiModel* timeline() { return &timelineModel; }
    RuntimeStatusUiModel* runtime() { return &runtimeModel; }
    NodeGraphModel* nodeGraph() { return &graphModel; }
    ConsoleModel* console() { return &consoleModel; }
private:
    ViewportUiModel viewportModel;
    HeatPaletteUiModel heatPaletteModel;
    TimelineUiModel timelineModel;
    RuntimeStatusUiModel runtimeModel;
    NodeGraphModel graphModel;
    ConsoleModel consoleModel;
};
