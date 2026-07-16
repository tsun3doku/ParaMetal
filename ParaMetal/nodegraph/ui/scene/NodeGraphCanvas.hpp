#pragma once

#include <QGraphicsView>
#include <QPoint>
#include <QPointF>

class QResizeEvent;
class NodeGraphNavHints;

class NodeGraphCanvas : public QGraphicsView {
    Q_OBJECT
public:
    explicit NodeGraphCanvas(QWidget* parent = nullptr);

signals:
    void requestCreateMenu(QPoint globalPos, QPointF scenePos, bool requireEmptySpace);
    void requestDeleteSelected();
    void requestCopySelected();
    void requestPaste();

public slots:
    void centerOnContent();

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;

private:
    void positionNavHints();

    bool isPanningWithMiddleMouse = false;
    QPoint lastPanPoint{};
    NodeGraphNavHints* navHints = nullptr;
};