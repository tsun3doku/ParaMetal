#pragma once

#include "NodeGraphTypes.hpp"

#include <string>
#include <vector>

class NodeGraph;

class NodeGraphValidator {
public:
    static bool socketTypesAccept(
        NodeGraphValueType srcResolved,
        const std::vector<NodeGraphValueType>& srcAccepted,
        NodeGraphValueType dstResolved,
        const std::vector<NodeGraphValueType>& dstAccepted);

    static bool socketAccepts(
        const NodeSocketSignature& srcSocket,
        NodeGraphValueType srcResolved,
        const NodeSocketSignature& dstSocket,
        NodeGraphValueType dstResolved);

    static bool canCreateConnection(
        const NodeGraph& document,
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        std::string& errorMessage,
        NodeGraphEdgeId ignoreExistingEdge = {});

private:
    static bool createsCycle(const NodeGraph& document, NodeGraphNodeId fromNode, NodeGraphNodeId toNode);
};
