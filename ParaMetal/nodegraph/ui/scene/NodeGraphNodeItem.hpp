#pragma once

#include "nodegraph/ui/scene/NodeGraphSceneStyle.hpp"
#include "nodegraph/NodeGraphTypes.hpp"

#include <QGraphicsObject>
#include <QPainterPath>

class NodeGraphNodeItem : public QGraphicsObject {
public:
    NodeGraphNodeItem(const NodeGraphNode& node, QGraphicsItem* parent = nullptr);

    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = nullptr) override;

    void setNodeState(const NodeGraphNodeState& state);
    void setHoveredState(bool hovered);

    const NodeGraphNodeState& nodeState() const;
    NodeGraphNodeId nodeId() const;
    nodegraphscene::NodeHitRegion hitRegionAt(const QPointF& localPos) const;
    QPointF inputSocketPosition(std::size_t index, std::size_t total) const;
    QPointF outputSocketPosition(std::size_t index, std::size_t total) const;

protected:
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent* event) override;

private:
    void drawCap(
        QPainter* painter,
        const QPainterPath& path,
        const QColor& activeColor,
        const QColor& inactiveColor,
        bool active,
        bool hovered) const;

    NodeGraphNodeId nodeIdValue{};
    NodeTypeId typeId;
    NodeGraphNodeState stateValue{};
    bool hoveredValue = false;
    nodegraphscene::NodeHitRegion hoverRegion = nodegraphscene::NodeHitRegion::None;
};
