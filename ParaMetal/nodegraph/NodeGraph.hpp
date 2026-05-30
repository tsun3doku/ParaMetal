#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class NodeGraphRegistry;

class NodeGraph {
public:
    explicit NodeGraph(const NodeGraphRegistry* registry = nullptr);

    NodeGraphNodeId addNode(const NodeTypeId& typeId, const std::string& title, float x, float y);
    bool removeNode(NodeGraphNodeId nodeId);
    bool moveNode(NodeGraphNodeId nodeId, float x, float y);
    bool setNodeDisplayEnabled(NodeGraphNodeId nodeId, bool enabled);
    bool setNodeFrozen(NodeGraphNodeId nodeId, bool frozen);
    bool setNodeParameter(NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter);
    bool appendSocket(NodeGraphNodeId nodeId, const NodeSocketSignature& socketSignature, NodeGraphSocketId* outSocketId = nullptr);

    bool addConnection(
        NodeGraphNodeId fromNode,
        NodeGraphSocketId fromSocket,
        NodeGraphNodeId toNode,
        NodeGraphSocketId toSocket,
        NodeGraphEdgeId* outEdgeId = nullptr);
    bool removeConnection(NodeGraphEdgeId edgeId);

    void clear();
    bool loadSerializedState(
        const NodeGraphState& state,
        uint32_t serializedNextNodeId,
        uint32_t serializedNextSocketId,
        uint32_t serializedNextEdgeId,
        std::string& errorMessage);
    void getNextIds(uint32_t& outNodeId, uint32_t& outSocketId, uint32_t& outEdgeId) const;

    const std::unordered_map<uint32_t, NodeGraphNode>& getNodes() const {
        return nodes;
    }

    const std::unordered_map<uint32_t, NodeGraphEdge>& getEdges() const {
        return edges;
    }

    const NodeGraphNode* findNode(NodeGraphNodeId nodeId) const;
    NodeGraphNode* findNode(NodeGraphNodeId nodeId);

    uint64_t getRevision() const {
        return revision;
    }

private:
    NodeGraphSocketId allocateSocketId();
    std::vector<NodeGraphSocket> buildSocketsFromInterface(const NodeTypeDefinition& definition, NodeGraphSocketDirection direction);
    void bumpRevision();

    uint32_t nextNodeId = 1;
    uint32_t nextSocketId = 1;
    uint32_t nextEdgeId = 1;
    uint64_t revision = 1;
    const NodeGraphRegistry* registry = nullptr;

    std::unordered_map<uint32_t, NodeGraphNode> nodes;
    std::unordered_map<uint32_t, NodeGraphEdge> edges;
};
