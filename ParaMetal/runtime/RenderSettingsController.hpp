#pragma once

#include "render/RenderSettings.hpp"
#include "scene/InputActions.hpp"

#include <QObject>

class RenderSettingsManager;

class RenderSettingsController : public QObject, public InputActionHandler {
    Q_OBJECT

public:
    using WireframeMode = app::WireframeMode;

    explicit RenderSettingsController(RenderSettingsManager* settingsManager = nullptr,
                                      QObject* parent = nullptr);

    void bind(RenderSettingsManager* settingsManager);

    void setWireframeMode(WireframeMode mode);
    void toggleGrid();

    app::RenderSettings getSnapshot() const;

signals:
    void settingsChanged();

private:
    void onWireframeToggleRequested() override;
    void onTimingOverlayToggleRequested() override;
    void onGridToggleRequested() override;

    RenderSettingsManager* settingsManager = nullptr;
};
