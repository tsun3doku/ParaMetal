#pragma once

#include "render/RenderSettings.hpp"

#include <mutex>

class RenderSettingsManager {
public:
    void setWireframeMode(app::WireframeMode mode);

    void toggleWireframeMode();
    void toggleTimingOverlay();
    void toggleGrid();

    app::RenderSettings getSnapshot() const;

private:
    mutable std::mutex mutex;
    app::RenderSettings settings;
};
