#include "NodeGraphCanvas.hpp"

#include <QContextMenuEvent>
#include <QCursor>
#include <QFrame>
#include <QKeyEvent>
#include <QKeySequence>
#include <QMouseEvent>
#include <QPainter>
#include <QScrollBar>
#include <QWheelEvent>

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
}

void NodeGraphCanvas::centerOnContent() {
    if (!scene()) {
        return;
    }
    const QRectF contentRect = scene()->itemsBoundingRect();
    centerOn(contentRect.center());
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
    constexpr qreal zoomInFactor = 1.15;
    constexpr qreal zoomOutFactor = 1.0 / zoomInFactor;
    if (event->angleDelta().y() > 0) {
        scale(zoomInFactor, zoomInFactor);
    } else if (event->angleDelta().y() < 0) {
        scale(zoomOutFactor, zoomOutFactor);
    }
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