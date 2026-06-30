#pragma once

#include "NodeGraphEdges.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <unordered_map>

struct NodeGraphState {
    uint64_t revision = 0;
    std::unordered_map<uint32_t, NodeGraphNode> nodes;
    NodeGraphEdges edges;

    const NodeGraphNode* node(NodeGraphNodeId id) const {
        auto it = nodes.find(id.value);
        return (it != nodes.end()) ? &it->second : nullptr;
    }
};
