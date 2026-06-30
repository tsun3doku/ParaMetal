#include "NodeGraphEdges.hpp"

#include <utility>

NodeGraphEdges::NodeGraphEdges(const NodeGraphEdges& other)
    : edgeById(other.edgeById),
      indexDirty(true) {
}

NodeGraphEdges& NodeGraphEdges::operator=(const NodeGraphEdges& other) {
    if (this == &other) {
        return *this;
    }
    edgeById = other.edgeById;
    incomingByInputSocket.clear();
    indexDirty = true;
    return *this;
}

NodeGraphEdges::NodeGraphEdges(NodeGraphEdges&& other) noexcept
    : edgeById(std::move(other.edgeById)),
      indexDirty(true) {
    other.markDirty();
}

NodeGraphEdges& NodeGraphEdges::operator=(NodeGraphEdges&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    edgeById = std::move(other.edgeById);
    incomingByInputSocket.clear();
    indexDirty = true;
    other.markDirty();
    return *this;
}

const NodeGraphEdge* NodeGraphEdges::find(NodeGraphEdgeId edgeId) const {
    const auto it = edgeById.find(edgeId.value);
    return it != edgeById.end() ? &it->second : nullptr;
}

const NodeGraphEdge* NodeGraphEdges::incomingEdge(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const {
    const std::vector<const NodeGraphEdge*> edges = incoming(toNode, toSocket);
    return edges.empty() ? nullptr : edges.front();
}

std::vector<const NodeGraphEdge*> NodeGraphEdges::incoming(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const {
    rebuildIndexIfDirty();
    const uint64_t inputKey = NodeSocketKey(toNode, toSocket).value;
    const auto it = incomingByInputSocket.find(inputKey);
    if (it == incomingByInputSocket.end()) {
        return {};
    }
    return it->second;
}

NodeSocketKey NodeGraphEdges::upstream(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const {
    const NodeGraphEdge* edge = incomingEdge(toNode, toSocket);
    return edge ? NodeSocketKey(edge->fromNode, edge->fromSocket) : NodeSocketKey{};
}

std::vector<NodeSocketKey> NodeGraphEdges::upstreams(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const {
    const std::vector<const NodeGraphEdge*> edges = incoming(toNode, toSocket);
    std::vector<NodeSocketKey> keys;
    keys.reserve(edges.size());
    for (const NodeGraphEdge* edge : edges) {
        if (edge) {
            keys.emplace_back(edge->fromNode, edge->fromSocket);
        }
    }
    return keys;
}

std::vector<NodeGraphNodeId> NodeGraphEdges::upstreamNodes(NodeGraphNodeId toNode) const {
    std::vector<NodeGraphNodeId> nodeIds;
    for (const auto& [id, edge] : edgeById) {
        (void)id;
        if (edge.toNode != toNode || !edge.fromNode.isValid()) {
            continue;
        }

        bool alreadyPresent = false;
        for (NodeGraphNodeId existingId : nodeIds) {
            if (existingId == edge.fromNode) {
                alreadyPresent = true;
                break;
            }
        }
        if (!alreadyPresent) {
            nodeIds.push_back(edge.fromNode);
        }
    }
    return nodeIds;
}

bool NodeGraphEdges::isConnected(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const {
    return incomingEdge(toNode, toSocket) != nullptr;
}

const NodeGraphEdge& NodeGraphEdges::at(uint32_t edgeId) const {
    return edgeById.at(edgeId);
}

std::size_t NodeGraphEdges::size() const {
    return edgeById.size();
}

bool NodeGraphEdges::empty() const {
    return edgeById.empty();
}

NodeGraphEdges::const_iterator NodeGraphEdges::begin() const {
    return edgeById.begin();
}

NodeGraphEdges::const_iterator NodeGraphEdges::end() const {
    return edgeById.end();
}

NodeGraphEdges::Store NodeGraphEdges::toMap() const {
    return edgeById;
}

void NodeGraphEdges::upsert(const NodeGraphEdge& edge) {
    edgeById[edge.id.value] = edge;
    markDirty();
}

bool NodeGraphEdges::remove(NodeGraphEdgeId edgeId) {
    const auto erased = edgeById.erase(edgeId.value);
    if (erased == 0) {
        return false;
    }
    markDirty();
    return true;
}

void NodeGraphEdges::clear() {
    edgeById.clear();
    markDirty();
}

void NodeGraphEdges::assign(const Store& edges) {
    edgeById = edges;
    markDirty();
}

void NodeGraphEdges::assign(Store&& edges) {
    edgeById = std::move(edges);
    markDirty();
}

void NodeGraphEdges::rebuildIndexIfDirty() const {
    if (!indexDirty) {
        return;
    }
    incomingByInputSocket.clear();
    incomingByInputSocket.reserve(edgeById.size() * 2);
    for (const auto& [id, edge] : edgeById) {
        (void)id;
        incomingByInputSocket[NodeSocketKey(edge.toNode, edge.toSocket).value].push_back(&edge);
    }
    indexDirty = false;
}

void NodeGraphEdges::markDirty() {
    incomingByInputSocket.clear();
    indexDirty = true;
}
