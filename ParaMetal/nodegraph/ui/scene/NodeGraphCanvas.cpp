#include "NodeGraphCanvas.hpp"

#include "nodegraph/ui/widgets/NodeGraphNavHints.hpp"
#include "nodegraph/ui/widgets/NodeGraphWidgetStyle.hpp"

#include <QContextMenuEvent>
#include <QCursor>
#include <QFrame>
#include <QKeyEvent>
#include <QKeySequence>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QScrollBar>
#include <QWheelEvent>

#include <algorithm>

NodeGraphCanvas::NodeGraphCanvas(QWidget* parent)
    : QGraphicsView(parent) {
    setDragMode(QGraphicsView::RubberBandDrag);
    setRubberBandSelectionMode(Qt::IntersectsItemShape);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setFrameShape(QFrame::NoFrame);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    setCursor(Qt::ArrowCursor);
    viewport()->setCursor(Qt::ArrowCursor);
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::TextAntialiasing, true);

    navHints = new NodeGraphNavHints(this);
    navHints->raise();
}

void NodeGraphCanvas::centerOnContent() {
    if (!scene()) {
        return;
    }
    const QRectF contentRect = scene()->itemsBoundingRect();
    centerOn(contentRect.center());
}

void NodeGraphCanvas::resizeEvent(QResizeEvent* event) {
    QGraphicsView::resizeEvent(event);
    positionNavHints();
}

void NodeGraphCanvas::positionNavHints() {
    if (!navHints) {
        return;
    }
    const QSize hint = navHints->sizeHint();
    navHints->setGeometry(
        nodegraphwidgets::navHintCanvasMargin,
        height() - hint.height() - nodegraphwidgets::navHintCanvasMargin,
        hint.width(),
        hint.height());
}

void NodeGraphCanvas::mousePressEvent(QMouseEvent* event) {
    if (event && event->button() == Qt::MiddleButton) {
        isPanningWithMiddleMouse = true;
        lastPanPoint = event->pos();
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void NodeGraphCanvas::mouseMoveEvent(QMouseEvent* event) {
    if (event && isPanningWithMiddleMouse) {
        const QPoint delta = event->pos() - lastPanPoint;
        lastPanPoint = event->pos();
        if (horizontalScrollBar()) {
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        }
        if (verticalScrollBar()) {
            verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        }
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void NodeGraphCanvas::mouseReleaseEvent(QMouseEvent* event) {
    if (event && event->button() == Qt::MiddleButton) {
        isPanningWithMiddleMouse = false;
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void NodeGraphCanvas::wheelEvent(QWheelEvent* event) {
    if (!event) {
        return;
    }

    const int wheelDelta = event->angleDelta().y();
    if (wheelDelta == 0) {
        event->accept();
        return;
    }

    const qreal requestedFactor = wheelDelta > 0
        ? nodegraphcanvas::wheelZoomFactor
        : 1.0 / nodegraphcanvas::wheelZoomFactor;
    const qreal currentZoom = transform().m11();
    const qreal targetZoom = std::clamp(
        currentZoom * requestedFactor,
        nodegraphcanvas::minimumZoom,
        nodegraphcanvas::maximumZoom);

    scale(targetZoom / currentZoom, targetZoom / currentZoom);
    event->accept();
}

void NodeGraphCanvas::contextMenuEvent(QContextMenuEvent* event) {
    if (!event) {
        return;
    }
    emit requestCreateMenu(event->globalPos(), mapToScene(event->pos()), true);
    event->accept();
}

void NodeGraphCanvas::keyPressEvent(QKeyEvent* event) {
    if (!event) {
        return;
    }
    if (event->matches(QKeySequence::Copy)) {
        emit requestCopySelected();
        event->accept();
        return;
    }
    if (event->matches(QKeySequence::Paste)) {
        emit requestPaste();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Delete) {
        emit requestDeleteSelected();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Tab) {
        QPoint anchorViewPos = mapFromGlobal(QCursor::pos());
        if (!viewport()->rect().contains(anchorViewPos)) {
            anchorViewPos = viewport()->rect().center();
        }
        emit requestCreateMenu(viewport()->mapToGlobal(anchorViewPos), mapToScene(anchorViewPos), false);
        event->accept();
        return;
    }
    QGraphicsView::keyPressEvent(event);
}
