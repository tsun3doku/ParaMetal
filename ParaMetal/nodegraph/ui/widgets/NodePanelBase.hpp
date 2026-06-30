#pragma once

#include "nodegraph/NodeGraphTypes.hpp"

#include <QWidget>

#include <functional>

class NodeGraph;

class NodePanelBase : public QWidget {
public:
    explicit NodePanelBase(QWidget* parent = nullptr);

    void bind(NodeGraph* bridge);
    void setNode(NodeGraphNodeId nodeId);
    void setStatusSink(std::function<void(const QString&)> sink);

protected:
    virtual void refreshFromNode() = 0;

    bool canEdit() const;
    bool loadCurrentNode(NodeGraphNode& outNode);
    void setStatus(const QString& text) const;
    NodeGraph* bridge() const;
    NodeGraphNodeId currentNodeId() const;
    void setSyncing(bool value);
    bool isSyncing() const;

    NodeGraph* bridgePtr = nullptr;
    NodeGraphNodeId nodeId{};
    bool syncing = false;

private:
    std::function<void(const QString&)> statusSink;
};
