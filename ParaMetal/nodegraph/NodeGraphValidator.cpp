#include "NodeGraphValidator.hpp"
#include "NodeGraph.hpp"
#include "NodeGraphTypes.hpp"
#include "NodeGraphUtils.hpp"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

bool NodeGraphValidator::socketTypesAccept(
    NodeGraphValueType srcResolved,
    const std::vector<NodeGraphValueType>& srcAccepted,
    NodeGraphValueType dstResolved,
    const std::vector<NodeGraphValueType>& dstAccepted) {

    if (srcResolved != NodeGraphValueType::None && dstResolved != NodeGraphValueType::None) {
        return srcResolved == dstResolved;
    }

    if (!srcAccepted.empty()) {
        if (dstResolved != NodeGraphValueType::None) {
            return std::find(srcAccepted.begin(), srcAccepted.end(), dstResolved) != srcAccepted.end();
        }
        if (!dstAccepted.empty()) {
            return std::any_of(srcAccepted.begin(), srcAccepted.end(), [&](NodeGraphValueType vt) {
                return std::find(dstAccepted.begin(), dstAccepted.end(), vt) != dstAccepted.end();
            });
        }
    }

    if (!dstAccepted.empty() && srcResolved != NodeGraphValueType::None) {
        return std::find(dstAccepted.begin(), dstAccepted.end(), srcResolved) != dstAccepted.end();
    }

    return srcResolved == dstResolved;
}

bool NodeGraphValidator::socketAccepts(
    const NodeSocketSignature& srcSocket,
    NodeGraphValueType srcResolved,
    const NodeSocketSignature& dstSocket,
    NodeGraphValueType dstResolved) {
    return socketTypesAccept(
        srcResolved,
        srcSocket.acceptedValueTypes,
        dstResolved,
        dstSocket.acceptedValueTypes);
}

bool NodeGraphValidator::canCreateConnection(
    const NodeGraph& document,
    NodeGraphNodeId fromNode,
    NodeGraphSocketId fromSocket,
    NodeGraphNodeId toNode,
    NodeGraphSocketId toSocket,
    std::string& errorMessage,
    NodeGraphEdgeId ignoreExistingEdge) {
    errorMessage.clear();

    if (!fromNode.isValid() || !toNode.isValid()) {
        errorMessage = "Connection uses an invalid node id.";
        return false;
    }

    if (fromNode == toNode) {
        errorMessage = "A node cannot connect to itself.";
        return false;
    }

    const NodeGraphNode* srcNode = document.findNode(fromNode);
    const NodeGraphNode* dstNode = document.findNode(toNode);
    if (!srcNode || !dstNode) {
        errorMessage = "Connection references a missing node.";
        return false;
    }

    const NodeGraphSocket* srcSocket = srcNode ? srcNode->output(fromSocket) : nullptr;
    const NodeGraphSocket* dstSocket = dstNode ? dstNode->input(toSocket) : nullptr;
    if (!srcSocket || !dstSocket) {
        errorMessage = "Connection references a missing input or output socket.";
        return false;
    }

    const NodeGraphValueType srcType = socketType(document, fromNode, fromSocket);
    const NodeGraphValueType dstType = socketType(document, toNode, toSocket);

    if (!socketAccepts(*srcSocket, srcType, *dstSocket, dstType)) {
        errorMessage = "Type mismatch: output '" + srcSocket->name +
            "' is '" + valueTypeToString(srcType) +
            "' but input '" + dstSocket->name + "' expects '" +
            valueTypeToString(dstSocket->acceptedValueTypes.empty() ? dstSocket->valueType : dstType) + "'.";
        return false;
    }

    for (const auto& [id, edge] : document.getEdges()) {
        if (ignoreExistingEdge.isValid() && edge.id == ignoreExistingEdge) {
            continue;
        }
        if (edge.fromNode == fromNode && edge.fromSocket == fromSocket && edge.toNode == toNode && edge.toSocket == toSocket) {
            errorMessage = "Connection already exists.";
            return false;
        }
        if (edge.toNode == toNode && edge.toSocket == toSocket && !dstSocket->variadic) {
            errorMessage = "Input socket already has a connection.";
            return false;
        }
    }

    if (createsCycle(document, fromNode, toNode)) {
        errorMessage = "Connection would introduce a cycle.";
        return false;
    }

    return true;
}

bool NodeGraphValidator::createsCycle(const NodeGraph& document, NodeGraphNodeId fromNode, NodeGraphNodeId toNode) {
    std::unordered_map<uint32_t, std::vector<uint32_t>> adjacency;
    for (const auto& [id, edge] : document.getEdges()) {
        adjacency[edge.fromNode.value].push_back(edge.toNode.value);
    }

    std::queue<uint32_t> pending;
    std::unordered_set<uint32_t> visited;
    pending.push(toNode.value);

    while (!pending.empty()) {
        const uint32_t node = pending.front();
        pending.pop();

        if (!visited.insert(node).second) {
            continue;
        }

        if (node == fromNode.value) {
            return true;
        }

        const auto adjIt = adjacency.find(node);
        if (adjIt == adjacency.end()) {
            continue;
        }

        for (uint32_t next : adjIt->second) {
            pending.push(next);
        }
    }

    return false;
}
