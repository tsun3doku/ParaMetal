#include "ViewportItem.hpp"

#include "ViewportRenderer.hpp"

#include <QtGui/QFocusEvent>
#include <QtGui/QHoverEvent>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtQuick/QQuickWindow>

ViewportItem::ViewportItem(QQuickItem* parent)
    : QQuickRhiItem(parent) {
    setFlag(ItemHasContents, true);
    setAcceptedMouseButtons(Qt::LeftButton | Qt::MiddleButton | Qt::RightButton);
    setAcceptHoverEvents(true);
}

void ViewportItem::bindRuntime(RuntimeSystems& runtime_, RuntimeNotifier& notifier_) {
    Q_ASSERT(!runtime && !notifier);
    runtime = &runtime_;
    notifier = &notifier_;
}

void ViewportItem::requestProbePicking(int nodeId, int probeIndex) {
    mailbox.requestProbePicking(nodeId, probeIndex);
    update();
}

void ViewportItem::requestExport(int nodeId) {
    mailbox.requestExport(nodeId);
    update();
}

void ViewportItem::requestExportProducts(int nodeId, qulonglong requestId, qulonglong graphRevision) {
    mailbox.requestExportProducts(nodeId, requestId, graphRevision);
    update();
}

void ViewportItem::requestWorldUnit(int unit) {
    mailbox.requestWorldUnit(unit);
    update();
}

void ViewportItem::requestWireframeMode(app::WireframeMode mode) { mailbox.requestWireframeMode(mode); update(); }
void ViewportItem::requestGridEnabled(bool enabled) { mailbox.requestGridEnabled(enabled); update(); }
void ViewportItem::requestBackgroundMode(app::BackgroundMode mode) { mailbox.requestBackgroundMode(mode); update(); }
void ViewportItem::requestNavigationCubeVisible(bool visible) { mailbox.requestNavigationCubeVisible(visible); update(); }
void ViewportItem::requestAxisLabelsVisible(bool visible) { mailbox.requestAxisLabelsVisible(visible); update(); }
void ViewportItem::requestFocusWorldOrigin() { mailbox.requestFocusWorldOrigin(); update(); }
void ViewportItem::requestProjectionMode(CameraProjectionMode mode) { mailbox.requestProjectionMode(mode); update(); }
void ViewportItem::requestCameraFov(float degrees) { mailbox.requestCameraFov(degrees); update(); }
void ViewportItem::requestCameraZoomSpeed(float speed) { mailbox.requestCameraZoomSpeed(speed); update(); }
void ViewportItem::requestCameraPanSpeed(float speed) { mailbox.requestCameraPanSpeed(speed); update(); }
void ViewportItem::requestTimelinePlaying(bool playing) { mailbox.requestTimelinePlaying(playing); update(); }
void ViewportItem::requestTimelineReset() { mailbox.requestTimelineReset(); update(); }
void ViewportItem::requestTimelineScrub(uint32_t frame) { mailbox.requestTimelineScrub(frame); update(); }
void ViewportItem::requestTimelineStep(int delta) { mailbox.requestTimelineStep(delta); update(); }
void ViewportItem::requestTimelineRange(uint32_t frameCount, float fps) { mailbox.requestTimelineRange(frameCount, fps); update(); }
void ViewportItem::requestSelection(int nodeId) { mailbox.requestSelection(nodeId); update(); }
void ViewportItem::requestHeatPaletteRange(float minimum, float maximum) { mailbox.requestHeatPaletteRange(minimum, maximum); update(); }
void ViewportItem::requestHeatPalette(int palette) { mailbox.requestHeatPalette(palette); update(); }
void ViewportItem::requestVelocityPaletteRange(float minimum, float maximum) { mailbox.requestVelocityPaletteRange(minimum, maximum); update(); }
void ViewportItem::requestVelocityPalette(int palette) { mailbox.requestVelocityPalette(palette); update(); }
void ViewportItem::applyViewportProjectState(const ProjectFile::Viewport& viewport) {
    mailbox.applyViewportProjectState(viewport);
    update();
}

void ViewportItem::requestCurrentViewportProjectState() {
    mailbox.requestCurrentViewportProjectState();
    update();
}

void ViewportItem::initializeGraph(const NodeGraphState& graphState) {
    mailbox.replaceGraphState(graphState);
    update();
}

void ViewportItem::queueGraphDelta(const NodeGraphDelta& delta) {
    mailbox.appendGraphDelta(delta);
    update();
}

QQuickRhiItemRenderer* ViewportItem::createRenderer() {
    Q_ASSERT(runtime && notifier);
    return runtime && notifier ? new ViewportRenderer(*runtime, *notifier, mailbox, *this) : nullptr;
}

void ViewportItem::keyPressEvent(QKeyEvent* event) {
    if (!event || event->isAutoRepeat()) {
        QQuickRhiItem::keyPressEvent(event);
        return;
    }

    WindowRuntimeState& state = mailbox.runtimeState();
    const Qt::Key key = static_cast<Qt::Key>(event->key());
    if (key == Qt::Key_Shift) {
        state.shiftPressed.store(true, std::memory_order_release);
    }

    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::Key;
    inputEvent.key = static_cast<int>(key);
    inputEvent.pressed = true;
    inputEvent.ctrlPressed = event->modifiers().testFlag(Qt::ControlModifier);
    state.pushInputEvent(inputEvent);
    event->accept();
}

void ViewportItem::keyReleaseEvent(QKeyEvent* event) {
    if (!event || event->isAutoRepeat()) {
        QQuickRhiItem::keyReleaseEvent(event);
        return;
    }

    WindowRuntimeState& state = mailbox.runtimeState();
    const Qt::Key key = static_cast<Qt::Key>(event->key());
    if (key == Qt::Key_Shift) {
        state.shiftPressed.store(false, std::memory_order_release);
    }

    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::Key;
    inputEvent.key = static_cast<int>(key);
    inputEvent.pressed = false;
    inputEvent.ctrlPressed = event->modifiers().testFlag(Qt::ControlModifier);
    state.pushInputEvent(inputEvent);
    event->accept();
}

void ViewportItem::mousePressEvent(QMouseEvent* event) {
    if (!event) {
        return;
    }

    WindowRuntimeState& state = mailbox.runtimeState();
    forceActiveFocus();
    const QPointF pixelPosition = devicePixelPosition(event->position());
    if (event->button() == Qt::MiddleButton) {
        state.middleButtonPressed.store(true, std::memory_order_release);
    }

    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::MousePress;
    inputEvent.button = static_cast<int>(event->button());
    inputEvent.shiftPressed = event->modifiers().testFlag(Qt::ShiftModifier);
    inputEvent.x = static_cast<float>(pixelPosition.x());
    inputEvent.y = static_cast<float>(pixelPosition.y());
    state.pushInputEvent(inputEvent);
    event->accept();
}

void ViewportItem::mouseReleaseEvent(QMouseEvent* event) {
    if (!event) {
        return;
    }

    WindowRuntimeState& state = mailbox.runtimeState();
    const QPointF pixelPosition = devicePixelPosition(event->position());
    if (event->button() == Qt::MiddleButton) {
        state.middleButtonPressed.store(false, std::memory_order_release);
    }

    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::MouseRelease;
    inputEvent.button = static_cast<int>(event->button());
    inputEvent.x = static_cast<float>(pixelPosition.x());
    inputEvent.y = static_cast<float>(pixelPosition.y());
    state.pushInputEvent(inputEvent);
    event->accept();
}

void ViewportItem::mouseMoveEvent(QMouseEvent* event) {
    if (!event) {
        return;
    }

    pushMouseMoveEvent(event->position());
    event->accept();
}

void ViewportItem::hoverMoveEvent(QHoverEvent* event) {
    if (!event) {
        return;
    }

    pushMouseMoveEvent(event->position());
    event->accept();
}

void ViewportItem::hoverLeaveEvent(QHoverEvent* event) {
    WindowRuntimeState& state = mailbox.runtimeState();
    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::MouseMove;
    inputEvent.x = -1.0f;
    inputEvent.y = -1.0f;
    state.pushInputEvent(inputEvent);
    if (event) {
        event->accept();
    }
}

void ViewportItem::wheelEvent(QWheelEvent* event) {
    if (!event) {
        return;
    }

    WindowRuntimeState& state = mailbox.runtimeState();
    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::Scroll;
    inputEvent.yOffset = static_cast<double>(event->angleDelta().y()) / 120.0;
    state.pushInputEvent(inputEvent);
    event->accept();
}

void ViewportItem::focusOutEvent(QFocusEvent* event) {
    WindowRuntimeState& state = mailbox.runtimeState();
    state.shiftPressed.store(false, std::memory_order_release);
    state.middleButtonPressed.store(false, std::memory_order_release);
    QQuickRhiItem::focusOutEvent(event);
}

QPointF ViewportItem::devicePixelPosition(const QPointF& position) const {
    const qreal dpr = window() ? window()->devicePixelRatio() : 1.0;
    return position * dpr;
}

void ViewportItem::pushMouseMoveEvent(const QPointF& position) {
    const QPointF pixelPosition = devicePixelPosition(position);
    WindowInputEvent inputEvent{};
    inputEvent.type = WindowInputEventType::MouseMove;
    inputEvent.x = static_cast<float>(pixelPosition.x());
    inputEvent.y = static_cast<float>(pixelPosition.y());
    mailbox.runtimeState().pushInputEvent(inputEvent);
}
