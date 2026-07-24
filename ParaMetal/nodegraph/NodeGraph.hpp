#pragma once

#include "NodeGraphCompiler.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphState.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class NodeGraph {
public:
    NodeGraph();

    NodeGraphNodeId addNode(const NodeTypeId& typeId, const std::string& title, float x, float y);
    bool removeNode(NodeGraphNodeId nodeId);
    bool moveNode(NodeGraphNodeId nodeId, float x, float y);
    bool getNode(NodeGraphNodeId nodeId, NodeGraphNode& outNode) const;
    bool toggleNodeFrozen(NodeGraphNodeId nodeId);
    bool toggleNodeDisplay(NodeGraphNodeId nodeId);
    bool setNodeParameter(NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter);
    bool setNodeParameters(NodeGraphNodeId nodeId, const std::vector<NodeGraphParamValue>& parameters);
    bool appendSocket(NodeGraphNodeId nodeId, const NodeSocketSignature& socketSignature, NodeGraphSocketId* outSocketId = nullptr);

    bool addConnection(
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        NodeGraphEdgeId* outEdgeId = nullptr);
    bool connectSockets(
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        std::string& errorMessage,
        bool replaceExistingInput = true);
    bool removeConnection(NodeGraphEdgeId edgeId);

    void clear();
    bool canExecute(std::string& reason) const;
    NodeGraphState state() const;
    bool consumeChanges(uint64_t& lastSeenRevision, NodeGraphDelta& outDelta) const;
    bool resolveGizmoTransformNode(uint64_t outputSocketKey, NodeGraphNodeId& outTransformNodeId) const;
    bool loadSerializedState(
        const NodeGraphState& state,
        uint32_t serializedNextNodeId,
        uint32_t serializedNextSocketId,
        uint32_t serializedNextEdgeId,
        std::string& errorMessage);
    void getNextIds(uint32_t& outNodeId, uint32_t& outSocketId, uint32_t& outEdgeId) const;

    NodeGraphRegistry& getRegistry() { return registry; }
    const NodeGraphRegistry& getRegistry() const { return registry; }

    std::unordered_map<uint32_t, NodeGraphNode> getNodes() const;
    std::unordered_map<uint32_t, NodeGraphEdge> getEdges() const;

    const NodeGraphNode* findNode(NodeGraphNodeId nodeId) const;
    NodeGraphNode* findNode(NodeGraphNodeId nodeId);

    const NodeGraphEdge* findEdge(NodeGraphEdgeId edgeId) const;
    const NodeGraphEdge* findIncomingEdge(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;

    uint64_t getRevision() const;

private:
    const NodeGraphNode* findNodeUnlocked(NodeGraphNodeId nodeId) const;
    NodeGraphNode* findNodeUnlocked(NodeGraphNodeId nodeId);
    const NodeGraphEdge* findEdgeUnlocked(NodeGraphEdgeId edgeId) const;
    const NodeGraphEdge* findIncomingEdgeUnlocked(NodeGraphNodeId toNode, NodeGraphSocketId toSocket) const;
    NodeGraphSocketId allocateSocketId();
    std::vector<NodeGraphSocket> buildSocketsFromInterface(const NodeTypeDefinition& definition, NodeGraphSocketDirection direction);
    void rebuildStateLocked();
    void pushChangesLocked(const std::vector<NodeGraphChange>& changes);
    void bumpRevision();

    mutable std::recursive_mutex mutex;
    NodeGraphRegistry registry;
    uint32_t nextNodeId = 1;
    uint32_t nextSocketId = 1;
    uint32_t nextEdgeId = 1;
    uint64_t revision = 0;

    std::unordered_map<uint32_t, NodeGraphNode> nodes;
    NodeGraphEdges edges;
    NodeGraphState graphState;
    std::vector<std::pair<uint64_t, NodeGraphChange>> changeLog;
};
