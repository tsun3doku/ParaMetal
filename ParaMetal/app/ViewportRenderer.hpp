#pragma once

#include <QtCore/QElapsedTimer>
#include <QtQuick/QQuickRhiItem>

#include "UiRuntimeTypes.hpp"

class RuntimeNotifier;
class RuntimeSystems;
class ViewportItem;
class ViewportMailbox;

class ViewportRenderer final : public QQuickRhiItemRenderer {
public:
    ViewportRenderer(
        RuntimeSystems& runtime,
        RuntimeNotifier& notifier,
        ViewportMailbox& mailbox,
        ViewportItem& viewportItem);
    ~ViewportRenderer() override;

protected:
    void initialize(QRhiCommandBuffer* commandBuffer) override;
    void synchronize(QQuickRhiItem* item) override;
    void render(QRhiCommandBuffer* commandBuffer) override;

private:
    void publishState();

    RuntimeSystems& runtime;
    RuntimeNotifier& notifier;
    ViewportMailbox& mailbox;
    ViewportItem& viewportItem;
    QElapsedTimer frameTimer;
    QElapsedTimer snapshotTimer;
    ViewportUiState publishedViewport{};
    bool publishedHeatPaletteVisible = false;
    bool publishedVelocityPaletteVisible = false;
    TimelineUiState publishedTimeline{};
    bool initialized = false;
    bool runtimeFresh = false;
    bool probePickingActiveLast = false;
    bool publishedModelSelected = false;
};
