#pragma once

#include "UiRuntimeTypes.hpp"
#include "mesh/MeshElementType.hpp"

#include <QObject>

class ViewportUiModel final : public QObject {
    Q_OBJECT
    Q_PROPERTY(int wireframeMode READ wireframeMode NOTIFY stateChanged)
    Q_PROPERTY(bool gridEnabled READ gridEnabled NOTIFY stateChanged)
    Q_PROPERTY(int backgroundMode READ backgroundMode NOTIFY stateChanged)
    Q_PROPERTY(bool navigationCubeVisible READ navigationCubeVisible NOTIFY stateChanged)
    Q_PROPERTY(bool axisLabelsVisible READ axisLabelsVisible NOTIFY stateChanged)
    Q_PROPERTY(int projectionMode READ projectionMode NOTIFY stateChanged)
    Q_PROPERTY(float cameraFov READ cameraFov NOTIFY stateChanged)
    Q_PROPERTY(float cameraZoomSpeed READ cameraZoomSpeed NOTIFY stateChanged)
    Q_PROPERTY(float cameraPanSpeed READ cameraPanSpeed NOTIFY stateChanged)
    Q_PROPERTY(float nearClip READ nearClip NOTIFY stateChanged)
    Q_PROPERTY(bool cullLinesOnly READ cullLinesOnly NOTIFY stateChanged)
    Q_PROPERTY(bool modelSelected READ modelSelected NOTIFY modelSelectionChanged)
    Q_PROPERTY(bool meshSelectionActive READ meshSelectionActive NOTIFY meshSelectionStateChanged)
    Q_PROPERTY(bool selectionCommitRequired READ selectionCommitRequired NOTIFY selectionCommitRequiredChanged)
    Q_PROPERTY(int meshSelectionCount READ meshSelectionCount NOTIFY meshSelectionStateChanged)
    Q_PROPERTY(int meshSelectionElementType READ meshSelectionElementType NOTIFY meshSelectionElementTypeChanged)

public:
    explicit ViewportUiModel(QObject* parent = nullptr) : QObject(parent) {}

    int wireframeMode() const { return static_cast<int>(state.wireframeMode); }
    bool gridEnabled() const { return state.gridEnabled; }
    int backgroundMode() const { return static_cast<int>(state.backgroundMode); }
    bool navigationCubeVisible() const { return state.navigationCubeVisible; }
    bool axisLabelsVisible() const { return state.axisLabelsVisible; }
    int projectionMode() const { return static_cast<int>(state.projectionMode); }
    float cameraFov() const { return state.baseFov; }
    float cameraZoomSpeed() const { return state.zoomSpeed; }
    float cameraPanSpeed() const { return state.panSpeed; }
    float nearClip() const { return state.nearClip; }
    bool cullLinesOnly() const { return state.cullLinesOnly; }
    bool modelSelected() const { return modelIsSelected; }
    bool meshSelectionActive() const { return selectionActive; }
    bool selectionCommitRequired() const { return commitRequired; }
    int meshSelectionCount() const { return selectionCount; }
    int meshSelectionElementType() const { return selectionElementType; }

    Q_INVOKABLE void setWireframeMode(int mode);
    Q_INVOKABLE void setGridEnabled(bool enabled);
    Q_INVOKABLE void setBackgroundMode(int mode);
    Q_INVOKABLE void setNavigationCubeVisible(bool visible);
    Q_INVOKABLE void setAxisLabelsVisible(bool visible);
    Q_INVOKABLE void focusWorldOrigin();
    Q_INVOKABLE void setProjectionMode(int mode);
    Q_INVOKABLE void setCameraFov(float degrees);
    Q_INVOKABLE void setCameraZoomSpeed(float speed);
    Q_INVOKABLE void setCameraPanSpeed(float speed);
    Q_INVOKABLE void setNearClip(float meters);
    Q_INVOKABLE void setCullLinesOnly(bool linesOnly);
    Q_INVOKABLE void setMeshSelectionActive(bool active);
    Q_INVOKABLE void setMeshSelectionElementType(int elementType);

public slots:
    void applyState(const ViewportUiState& updated);
    void applyModelSelection(bool selected);
    void applySubselectionState(
        bool active,
        int selectedCount,
        int elementType,
        bool commitRequired);

signals:
    void stateChanged();
    void wireframeModeRequested(app::WireframeMode mode);
    void gridEnabledRequested(bool enabled);
    void backgroundModeRequested(app::BackgroundMode mode);
    void navigationCubeVisibleRequested(bool visible);
    void axisLabelsVisibleRequested(bool visible);
    void focusWorldOriginRequested();
    void projectionModeRequested(CameraProjectionMode mode);
    void cameraFovRequested(float degrees);
    void cameraZoomSpeedRequested(float speed);
    void cameraPanSpeedRequested(float speed);
    void nearClipRequested(float meters);
    void cullLinesOnlyRequested(bool linesOnly);
    void meshSelectionRequested(bool active);
    void modelSelectionChanged();
    void meshSelectionStateChanged();
    void selectionCommitRequiredChanged();
    void meshSelectionElementTypeChanged();
    void meshSelectionElementTypeRequested(int elementType);

private:
    ViewportUiState state{};
    bool modelIsSelected = false;
    bool selectionActive = false;
    bool commitRequired = false;
    int selectionCount = 0;
    int selectionElementType = static_cast<int>(MeshElementType::Face);
};
