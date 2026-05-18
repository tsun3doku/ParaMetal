#pragma once

#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphEditor.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <QGraphicsScene>
#include <QPointF>
#include <QString>

class QGraphicsSceneMouseEvent;
class QGraphicsItem;
class QGraphicsPathItem;
class QGraphicsSimpleTextItem;
class QPainter;
class QRectF;
class NodeGraphNodeItem;
class NodeGraphSocketItem;

class NodeGraphScene : public QGraphicsScene {
    Q_OBJECT
public:
    explicit NodeGraphScene(QObject* parent = nullptr);

    void setBridge(NodeGraphBridge* bridge);
    void applyPendingChanges();
    void setSelectedNode(NodeGraphNodeId nodeId);
    void clearNodeSelection();
    bool copySelectedNodes();
    bool pasteCopiedNodes();
    bool removeSelectedNodes();

signals:
    void nodeActivated(NodeGraphNodeId nodeId);
    void nodeSelectionChanged(NodeGraphNodeId nodeId);
    void statusReported(const QString& text);
    void graphPopulated();

protected:
    void drawBackground(QPainter* painter, const QRectF& rect) override;
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;

private:
    static constexpr int NodeIdRole = 0;
    static constexpr int EdgeIdRole = 1;
    static constexpr int EdgeFromSocketRole = 6;
    static constexpr int EdgeToSocketRole = 7;
    static constexpr int EdgeBaseColorRole = 8;

    NodeGraphBridge* bridge = nullptr;
    NodeGraphEditor editor;
    uint64_t lastSeenRevision = 0;
    QGraphicsPathItem* activeDragLine = nullptr;
    QPointF activeDragStartPos{};
    NodeGraphNodeId activeDragFromNode{};
    NodeGraphSocketId activeDragFromSocket{};
    NodeGraphSocketDirection activeDragFromDirection = NodeGraphSocketDirection::Output;

    bool isDraggingConnection = false;
    bool suppressNodeActivationOnRelease = false;

    std::unordered_map<uint32_t, NodeGraphNodeItem*> nodeItemsById;
    std::unordered_map<uint32_t, QGraphicsPathItem*> edgeItemsById;
    std::unordered_map<uint32_t, NodeGraphSocketItem*> inputSocketItemsBySocket;
    std::unordered_map<uint32_t, NodeGraphSocketItem*> outputSocketItemsBySocket;

    NodeGraphNodeId hoveredNodeId{};
    NodeGraphEdgeId hoveredEdgeId{};
    NodeGraphSocketId hoveredSocketId{};

    std::vector<NodeGraphEditor::CopiedNode> copiedNodes;
    std::vector<NodeGraphEditor::CopiedEdge> copiedEdges;
    uint32_t pasteGeneration = 0;
    bool suppressSelectionChangedNotifications = false;

    void buildFromState(const NodeGraphState& state);
    void applyDelta(const NodeGraphDelta& delta);
    void clearSceneState();

    NodeGraphNodeItem* createNodeItem(const NodeGraphNode& node);
    void removeNodeItem(NodeGraphNodeId nodeId);
    void removeEdgesForNode(NodeGraphNodeId nodeId);
    QGraphicsPathItem* createEdgeItem(const NodeGraphEdge& edge);
    void removeEdgeItem(NodeGraphEdgeId edgeId);

    void syncNodePositionsToBridge();
    void clearActiveDragLine();
    void updateEdgePathsFromCurrentLayout();
    void updateHoverState(const QPointF& scenePos);
    void notifySelectedNodeChanged();
    void selectNodesById(const std::vector<NodeGraphNodeId>& nodeIds);
    std::vector<NodeGraphNodeId> selectedTopLevelNodeIds() const;
    NodeGraphNodeId selectedSingleNodeId() const;
    static void setNodeHovered(NodeGraphNodeItem* item, bool hovered);
    static void setEdgeHovered(QGraphicsPathItem* item, bool hovered);
    static void setSocketHovered(NodeGraphSocketItem* item, bool hovered);
    static NodeGraphEdgeId itemEdgeId(const QGraphicsItem* item);
    static bool extractSocketFromItem(const QGraphicsItem* item, NodeGraphNodeId& outNodeId, NodeGraphSocketId& outSocketId, NodeGraphSocketDirection& outDirection);
    bool socketAtScenePos(const QPointF& scenePos, NodeGraphNodeId& outNodeId, NodeGraphSocketId& outSocketId, NodeGraphSocketDirection& outDirection) const;
    bool handleNodeCapClick(const QPointF& scenePos);
    void reportStatus(const QString& text);
    static NodeGraphNodeId itemNodeId(const QGraphicsItem* item);
};
