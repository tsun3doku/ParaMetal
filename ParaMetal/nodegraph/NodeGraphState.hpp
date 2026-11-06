#pragma once

#include "NodeGraphEdges.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <unordered_map>

struct NodeGraphState {
    uint64_t revision = 0;
    uint32_t nextNodeId = 1;
    uint32_t nextSocketId = 1;
    uint32_t nextEdgeId = 1;
    std::unordered_map<uint32_t, NodeGraphNode> nodes;
    NodeGraphEdges edges;

    const NodeGraphNode* node(NodeGraphNodeId id) const {
        auto it = nodes.find(id.value);
        return (it != nodes.end()) ? &it->second : nullptr;
    }
};


inline void applyNodeGraphChange(NodeGraphState& state, const NodeGraphChange& change) {
    switch (change.type) {
    case NodeGraphChangeType::Reset:
        state.nodes.clear();
        state.edges.clear();
        break;
    case NodeGraphChangeType::NodeUpsert:
        state.nodes[change.node.id.value] = change.node;
        break;
    case NodeGraphChangeType::NodeRemoved:
        state.nodes.erase(change.nodeId.value);
        state.edges.removeIncident(change.nodeId);
        break;
    case NodeGraphChangeType::EdgeUpsert:
        state.edges.upsert(change.edge);
        break;
    case NodeGraphChangeType::EdgeRemoved:
        state.edges.remove(change.edgeId);
        break;
    }
}

inline bool applyNodeGraphDelta(NodeGraphState& state, const NodeGraphDelta& delta) {
    if (delta.fromRevision != state.revision) return false;
    for (const NodeGraphChange& change : delta.changes) applyNodeGraphChange(state, change);
    state.revision = delta.toRevision;
    return true;
}
