#pragma once

#include "NodeGraph.hpp"

#include <string>
#include <vector>

class NodeGraphTypeRegistry;

class NodeGraphValidator {
public:
    static bool canCreateConnection(
        const NodeGraph& document,
        const NodeGraphTypeRegistry& typeRegistry,
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        std::string& errorMessage,
        NodeGraphEdgeId ignoreExistingEdge = {});

private:
    static bool wouldIntroduceCycle(const NodeGraph& document, NodeGraphNodeId fromNode, NodeGraphNodeId toNode);
};
