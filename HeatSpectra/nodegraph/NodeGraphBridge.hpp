#pragma once

#include "NodeGraph.hpp"
#include "NodeGraphCompiler.hpp"
#include "NodeGraphRegistry.hpp"

#include <mutex>
#include <string>
#include <utility>
#include <vector>

class NodeGraphBridge {
public:
    NodeGraphBridge();

    void clear();

    NodeGraphNodeId addNode(const NodeTypeId& typeId, const std::string& title, float x, float y);
    bool removeNode(NodeGraphNodeId nodeId);
    bool moveNode(NodeGraphNodeId nodeId, float x, float y);
    bool getNode(NodeGraphNodeId nodeId, NodeGraphNode& outNode) const;
    bool setNodeDisplayEnabled(NodeGraphNodeId nodeId, bool enabled);
    bool setNodeFrozen(NodeGraphNodeId nodeId, bool frozen);
    bool setNodeParameter(NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter);

    bool connectSockets(
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        std::string& errorMessage,
        bool replaceExistingInput = true);
    bool removeConnection(NodeGraphEdgeId edgeId);

    bool canExecute(std::string& reason) const;

    NodeGraphState state() const;
    NodeGraphRegistry& getRegistry() { return registry; }
    const NodeGraphRegistry& getRegistry() const { return registry; }
    bool resolveGizmoTransformNode(uint64_t outputSocketKey, NodeGraphNodeId& outTransformNodeId) const;
    bool consumeChanges(uint64_t& lastSeenRevision, NodeGraphDelta& outDelta) const;

private:
    void rebuildStateLocked();
    void pushChangesLocked(const std::vector<NodeGraphChange>& changes);

    mutable std::mutex mutex;
    NodeGraphRegistry registry;
    NodeGraph document;
    NodeGraphState graphState;
    std::vector<std::pair<uint64_t, NodeGraphChange>> changeLog;
    uint64_t changeRevision = 0;
};
