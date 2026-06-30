#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <vector>

class NodeGraphEdges {
public:
    using Store = std::unordered_map<uint32_t, NodeGraphEdge>;
    using const_iterator = Store::const_iterator;

    NodeGraphEdges() = default;
    NodeGraphEdges(const NodeGraphEdges& other);
    NodeGraphEdges& operator=(const NodeGraphEdges& other);
    NodeGraphEdges(NodeGraphEdges&& other) noexcept;
    NodeGraphEdges& operator=(NodeGraphEdges&& other) noexcept;

    const NodeGraphEdge* find(NodeGraphEdgeId edgeId) const;
    const NodeGraphEdge* incomingEdge(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;
    std::vector<const NodeGraphEdge*> incoming(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;
    NodeSocketKey upstream(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;
    std::vector<NodeSocketKey> upstreams(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;
    std::vector<NodeGraphNodeId> upstreamNodes(NodeGraphNodeId toNode) const;
    bool isConnected(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;

    const NodeGraphEdge& at(uint32_t edgeId) const;
    std::size_t size() const;
    bool empty() const;
    const_iterator begin() const;
    const_iterator end() const;
    Store toMap() const;

    void upsert(const NodeGraphEdge& edge);
    bool remove(NodeGraphEdgeId edgeId);
    void clear();
    void assign(const Store& edges);
    void assign(Store&& edges);

private:
    void rebuildIndexIfDirty() const;
    void markDirty();

    Store edgeById;
    mutable std::unordered_map<uint64_t, std::vector<const NodeGraphEdge*>> incomingByInputSocket;
    mutable bool indexDirty = true;
};
