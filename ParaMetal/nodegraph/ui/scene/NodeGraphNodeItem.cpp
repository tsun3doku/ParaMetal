#include "NodeGraphNodeItem.hpp"

#include "nodegraph/NodeGraphIconRegistry.hpp"

#include <QCursor>
#include <QGraphicsSceneHoverEvent>
#include <QLinearGradient>
#include <QPainter>
#include <QPen>
#include <QRadialGradient>
#include <QStyleOptionGraphicsItem>

#include <algorithm>

static QColor blendColors(const QColor& a, const QColor& b, qreal t) {
    const qreal clamped = std::clamp(t, 0.0, 1.0);
    return QColor(
        static_cast<int>(a.red() + (b.red() - a.red()) * clamped),
        static_cast<int>(a.green() + (b.green() - a.green()) * clamped),
        static_cast<int>(a.blue() + (b.blue() - a.blue()) * clamped),
        static_cast<int>(a.alpha() + (b.alpha() - a.alpha()) * clamped));
}

NodeGraphNodeItem::NodeGraphNodeItem(const NodeGraphNode& node, QGraphicsItem* parent)
    : QGraphicsObject(parent),
      nodeIdValue(node.id),
      typeId(node.typeId),
      stateValue(node.state) {
    setAcceptHoverEvents(true);
    setAcceptedMouseButtons(Qt::LeftButton);
    setCacheMode(QGraphicsItem::DeviceCoordinateCache);
}

QRectF NodeGraphNodeItem::boundingRect() const {
    const qreal outlinePad = nodegraphscene::selectedBorderWidth * 0.5 + nodegraphscene::boundingOutlinePad;
    const qreal topPad = nodegraphscene::socketRadius + nodegraphscene::boundingSocketPad;
    return nodegraphscene::nodeRect().adjusted(-outlinePad, -topPad, outlinePad, outlinePad);
}

QPainterPath NodeGraphNodeItem::shape() const {
    QPainterPath path;
    path.addRoundedRect(nodegraphscene::nodeRect(), nodegraphscene::roundCorners, nodegraphscene::roundCorners);
    return path;
}

void NodeGraphNodeItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) {
    Q_UNUSED(widget);
    if (!painter || !option) {
        return;
    }

    const QRectF shellRect = nodegraphscene::outerFrameRect();
    const QRectF centerRect = nodegraphscene::centerPanelRect();
    const bool isSelected = (option->state & QStyle::State_Selected) != 0;
    QPainterPath shellPath;
    shellPath.addRoundedRect(shellRect, nodegraphscene::roundCorners, nodegraphscene::roundCorners);

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);

    painter->setPen(Qt::NoPen);
    painter->setBrush(QColor(0, 0, 0, nodegraphscene::nodeShadowAlpha));
    painter->drawRoundedRect(
        shellRect.translated(nodegraphscene::nodeShadowOffsetX, nodegraphscene::nodeShadowOffsetY),
        nodegraphscene::roundCorners, nodegraphscene::roundCorners);

    const QColor shellStroke = isSelected
        ? nodegraphscene::selectedBorderColor()
        : nodegraphscene::nodeShellOutlineColor();
    const qreal shellStrokeWidth = isSelected
        ? nodegraphscene::selectedBorderWidth
        : nodegraphscene::nodeOuterBorderWidth;

    painter->setBrush(QBrush(nodegraphscene::nodeShellColor()));
    painter->setPen(Qt::NoPen);
    painter->drawPath(shellPath);

    painter->save();
    painter->setClipPath(shellPath);

    drawCap(
        painter,
        nodegraphscene::leftCapPath(),
        nodegraphscene::frozenCapActiveColor(),
        nodegraphscene::frozenCapInactiveColor(),
        stateValue.isFrozen(),
        hoverRegion == nodegraphscene::NodeHitRegion::LeftCap);
    drawCap(
        painter,
        nodegraphscene::rightCapPath(),
        nodegraphscene::displayCapActiveColor(),
        nodegraphscene::displayCapInactiveColor(),
        stateValue.isPrimaryDisplay(),
        hoverRegion == nodegraphscene::NodeHitRegion::RightCap);

    QLinearGradient centerGradient(centerRect.topLeft(), centerRect.bottomLeft());
    centerGradient.setColorAt(0.0, nodegraphscene::nodeCenterFillColor());
    centerGradient.setColorAt(1.0, nodegraphscene::nodeCenterFillColor().darker(nodegraphscene::centerGradientDarken));
    painter->setBrush(centerGradient);
    painter->setPen(Qt::NoPen);
    painter->drawPath(nodegraphscene::centerPanelPath());

    painter->setPen(QPen(nodegraphscene::nodeShellOutlineColor(), nodegraphscene::capBorderWidth));
    painter->drawLine(QLineF(centerRect.topLeft(), centerRect.bottomLeft()));
    painter->drawLine(QLineF(centerRect.topRight(), centerRect.bottomRight()));
    painter->restore();

    painter->setPen(QPen(shellStroke, shellStrokeWidth));
    painter->setBrush(Qt::NoBrush);
    painter->drawPath(shellPath);

    const QRectF iconBounds = nodegraphscene::iconRect();
    const qreal devicePixelRatio = painter->device() ? painter->device()->devicePixelRatioF() : 1.0;
    const qreal zoomLevel = option->levelOfDetailFromTransform(painter->worldTransform());
    const qreal targetPixelWidth = iconBounds.width() * zoomLevel * devicePixelRatio;
    const QPixmap icon = NodeGraphIconRegistry::nodePixmapForType(typeId, targetPixelWidth);
    if (!icon.isNull()) {
        const QSizeF iconSize = icon.deviceIndependentSize();
        const qreal scale = std::min(iconBounds.width() / iconSize.width(), iconBounds.height() / iconSize.height());
        const QSizeF scaledSize(iconSize.width() * scale, iconSize.height() * scale);
        const QRectF scaledRect(
            iconBounds.center().x() - scaledSize.width() * 0.5,
            iconBounds.center().y() - scaledSize.height() * 0.5,
            scaledSize.width(),
            scaledSize.height());
        painter->setRenderHint(QPainter::SmoothPixmapTransform, true);
        painter->drawPixmap(scaledRect.toAlignedRect(), icon);
    }

    painter->restore();
}

void NodeGraphNodeItem::setNodeState(const NodeGraphNodeState& state) {
    if (stateValue == state) {
        return;
    }

    stateValue = state;
    update();
}

void NodeGraphNodeItem::setHoveredState(bool hovered) {
    if (hoveredValue == hovered) {
        return;
    }

    hoveredValue = hovered;
    update();
}

const NodeGraphNodeState& NodeGraphNodeItem::nodeState() const {
    return stateValue;
}

NodeGraphNodeId NodeGraphNodeItem::nodeId() const {
    return nodeIdValue;
}

nodegraphscene::NodeHitRegion NodeGraphNodeItem::hitRegionAt(const QPointF& localPos) const {
    return nodegraphscene::hitRegionAt(localPos);
}

QPointF NodeGraphNodeItem::inputSocketPosition(std::size_t index, std::size_t total) const {
    return nodegraphscene::inputSocketPosition(index, total);
}

QPointF NodeGraphNodeItem::outputSocketPosition(std::size_t index, std::size_t total) const {
    return nodegraphscene::outputSocketPosition(index, total);
}

void NodeGraphNodeItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event) {
    if (!event) {
        return;
    }

    const nodegraphscene::NodeHitRegion region = hitRegionAt(event->pos());
    if (hoverRegion != region) {
        hoverRegion = region;
        update();
    }
    setCursor((region == nodegraphscene::NodeHitRegion::LeftCap || region == nodegraphscene::NodeHitRegion::RightCap)
        ? Qt::PointingHandCursor
        : Qt::ArrowCursor);
    QGraphicsObject::hoverMoveEvent(event);
}

void NodeGraphNodeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent* event) {
    if (hoverRegion != nodegraphscene::NodeHitRegion::None) {
        hoverRegion = nodegraphscene::NodeHitRegion::None;
        update();
    }
    unsetCursor();
    QGraphicsObject::hoverLeaveEvent(event);
}

void NodeGraphNodeItem::drawCap(
    QPainter* painter,
    const QPainterPath& path,
    const QColor& activeColor,
    const QColor& inactiveColor,
    bool active,
    bool hovered) const {
    if (!painter) {
        return;
    }

    const QRectF rect = path.boundingRect();
    const QColor fill = hovered ? blendColors(inactiveColor, activeColor, 0.5) : (active ? activeColor : inactiveColor);

    if (active) {
        QRadialGradient glow(rect.center(), std::max(rect.width(), rect.height()) * nodegraphscene::capGlowRadiusMultiplier);
        QColor start = activeColor;
        start.setAlpha(nodegraphscene::capGlowAlpha);
        QColor end = activeColor;
        end.setAlpha(0);
        glow.setColorAt(0.0, start);
        glow.setColorAt(1.0, end);
        painter->setPen(Qt::NoPen);
        painter->setBrush(glow);
        painter->drawEllipse(rect.center(), nodegraphscene::capGlowRadius, nodegraphscene::capGlowRadius);
    }

    QLinearGradient fillGradient(rect.topLeft(), rect.bottomLeft());
    fillGradient.setColorAt(0.0, fill.lighter(active ? nodegraphscene::capActiveLighten : 100));
    fillGradient.setColorAt(1.0, fill.darker(active ? nodegraphscene::capActiveDarken : nodegraphscene::capInactiveDarken));
    painter->setPen(Qt::NoPen);
    painter->setBrush(fillGradient);
    painter->drawPath(path);
}
