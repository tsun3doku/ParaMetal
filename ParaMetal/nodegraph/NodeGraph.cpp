#include "NodeGraph.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeGraphValidator.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

NodeGraph::NodeGraph(const NodeGraphRegistry* reg) : registry(reg) {}

namespace {

const NodeGraphSocket* findSocket(
    const NodeGraphNode& node,
    NodeGraphSocketId socketId,
    NodeGraphSocketDirection direction) {
    const std::vector<NodeGraphSocket>& sockets =
        direction == NodeGraphSocketDirection::Input ? node.inputs : node.outputs;
    for (const NodeGraphSocket& socket : sockets) {
        if (socket.id == socketId && socket.direction == direction) {
            return &socket;
        }
    }
    return nullptr;
}

bool patchSocketIds(
    std::vector<NodeGraphSocket>& rebuiltSockets,
    const std::vector<NodeGraphSocket>& savedSockets,
    std::string& errorMessage) {
    if (rebuiltSockets.size() != savedSockets.size()) {
        errorMessage = "Saved node socket count does not match current node definition.";
        return false;
    }

    for (std::size_t i = 0; i < rebuiltSockets.size(); ++i) {
        const NodeGraphSocket& savedSocket = savedSockets[i];
        NodeGraphSocket& rebuiltSocket = rebuiltSockets[i];
        if (rebuiltSocket.name != savedSocket.name ||
            rebuiltSocket.valueType != savedSocket.valueType ||
            rebuiltSocket.direction != savedSocket.direction ||
            rebuiltSocket.contract.producedPayloadType != savedSocket.contract.producedPayloadType ||
            rebuiltSocket.variadic != savedSocket.variadic ||
            !savedSocket.id.isValid()) {
            errorMessage = "Saved node socket metadata does not match current node definition.";
            return false;
        }
        rebuiltSocket.id = savedSocket.id;
    }

    return true;
}

uint32_t maxSocketId(const NodeGraphNode& node) {
    uint32_t maxId = 0;
    for (const NodeGraphSocket& socket : node.inputs) {
        maxId = std::max(maxId, socket.id.value);
    }
    for (const NodeGraphSocket& socket : node.outputs) {
        maxId = std::max(maxId, socket.id.value);
    }
    return maxId;
}

} // namespace

NodeGraphNodeId NodeGraph::addNode(const NodeTypeId& typeId, const std::string& title, float x, float y) {
    if (!registry) {
        return {};
    }
    const NodeTypeDefinition* definition = registry->findNodeType(typeId);
    if (!definition) {
        return {};
    }

    NodeGraphNode node{};
    node.id = NodeGraphNodeId{nextNodeId++};
    node.typeId = definition->id;
    node.category = definition->category;
    node.title = title.empty() ? definition->displayName : title;
    node.x = x;
    node.y = y;
    node.displayEnabled = false;
    node.inputs = buildSocketsFromInterface(*definition, NodeGraphSocketDirection::Input);
    node.outputs = buildSocketsFromInterface(*definition, NodeGraphSocketDirection::Output);
    for (const NodeGraphParamDefinition& parameter : definition->parameters) {
        node.parameters.push_back(makeNodeGraphParamValue(parameter));
    }

    NodeGraphNodeId id = node.id;
    nodes[id.value] = std::move(node);
    bumpRevision();
    return id;
}

bool NodeGraph::removeNode(NodeGraphNodeId nodeId) {
    auto it = nodes.find(nodeId.value);
    if (it == nodes.end()) {
        return false;
    }

    nodes.erase(it);
    for (auto edgeIt = edges.begin(); edgeIt != edges.end(); ) {
        if (edgeIt->second.fromNode == nodeId || edgeIt->second.toNode == nodeId) {
            edgeIt = edges.erase(edgeIt);
        } else {
            ++edgeIt;
        }
    }
    bumpRevision();
    return true;
}

bool NodeGraph::moveNode(NodeGraphNodeId nodeId, float x, float y) {
    NodeGraphNode* node = findNode(nodeId);
    if (!node) {
        return false;
    }

    if (std::fabs(node->x - x) < 0.01f && std::fabs(node->y - y) < 0.01f) {
        return false;
    }

    node->x = x;
    node->y = y;
    bumpRevision();
    return true;
}

bool NodeGraph::setNodeDisplayEnabled(NodeGraphNodeId nodeId, bool enabled) {
    NodeGraphNode* node = findNode(nodeId);
    if (!node) {
        return false;
    }

    bool changed = false;
    if (enabled) {
        for (auto& [id, candidate] : nodes) {
            const bool shouldBeDisplayed = (candidate.id == nodeId);
            if (candidate.displayEnabled == shouldBeDisplayed) {
                continue;
            }

            candidate.displayEnabled = shouldBeDisplayed;
            changed = true;
        }
    } else if (node->displayEnabled) {
        node->displayEnabled = false;
        changed = true;
    }

    if (!changed) {
        return false;
    }

    bumpRevision();
    return true;
}

bool NodeGraph::setNodeFrozen(NodeGraphNodeId nodeId, bool frozen) {
    NodeGraphNode* node = findNode(nodeId);
    if (!node || node->frozen == frozen) {
        return false;
    }

    node->frozen = frozen;
    bumpRevision();
    return true;
}

bool NodeGraph::setNodeParameter(NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter) {
    NodeGraphNode* node = findNode(nodeId);
    if (!node) {
        return false;
    }

    const NodeTypeDefinition* definition = registry ? registry->findNodeType(node->typeId) : nullptr;
    if (!definition) {
        return false;
    }

    const NodeGraphParamDefinition* parameterDefinition = findNodeParamDefinition(*definition, parameter.id);
    NodeGraphParamValue normalizedParameter = parameter;
    if (!parameterDefinition ||
        !normalizeNodeGraphParamValue(*parameterDefinition, normalizedParameter) ||
        !validateNodeGraphParamValue(*parameterDefinition, normalizedParameter)) {
        return false;
    }

    NodeGraphParamValue* existingParameter = findNodeParamValue(*node, parameter.id);
    if (existingParameter) {
        *existingParameter = std::move(normalizedParameter);
    } else {
        node->parameters.push_back(std::move(normalizedParameter));
    }

    bumpRevision();
    return true;
}

bool NodeGraph::appendSocket(
    NodeGraphNodeId nodeId,
    const NodeSocketSignature& socketSignature,
    NodeGraphSocketId* outSocketId) {
    NodeGraphNode* node = findNode(nodeId);
    if (!node) {
        return false;
    }

    NodeGraphSocket socket{};
    socket.id = allocateSocketId();
    socket.name = socketSignature.name;
    socket.valueType = socketSignature.valueType;
    socket.direction = socketSignature.direction;
    socket.contract = socketSignature.contract;

    if (socket.direction == NodeGraphSocketDirection::Input) {
        node->inputs.push_back(socket);
    } else {
        node->outputs.push_back(socket);
    }

    if (outSocketId) {
        *outSocketId = socket.id;
    }

    bumpRevision();
    return true;
}

bool NodeGraph::addConnection(
    NodeGraphNodeId fromNode,
    NodeGraphSocketId fromSocket,
    NodeGraphNodeId toNode,
    NodeGraphSocketId toSocket,
    NodeGraphEdgeId* outEdgeId) {
    NodeGraphEdge edge{};
    edge.id = NodeGraphEdgeId{nextEdgeId++};
    edge.fromNode = fromNode;
    edge.fromSocket = fromSocket;
    edge.toNode = toNode;
    edge.toSocket = toSocket;

    edges[edge.id.value] = edge;
    bumpRevision();

    if (outEdgeId) {
        *outEdgeId = edge.id;
    }
    return true;
}

bool NodeGraph::removeConnection(NodeGraphEdgeId edgeId) {
    auto it = edges.find(edgeId.value);
    if (it == edges.end()) {
        return false;
    }

    edges.erase(it);
    bumpRevision();
    return true;
}

void NodeGraph::clear() {
    nodes.clear();
    edges.clear();
    nextNodeId = 1;
    nextSocketId = 1;
    nextEdgeId = 1;
    bumpRevision();
}

bool NodeGraph::loadSerializedState(
    const NodeGraphState& state,
    uint32_t serializedNextNodeId,
    uint32_t serializedNextSocketId,
    uint32_t serializedNextEdgeId,
    std::string& errorMessage) {
    if (!registry) {
        errorMessage = "Node graph registry is not available.";
        return false;
    }

    NodeGraph candidate(registry);

    uint32_t maxNodeId = 0;
    uint32_t maxRestoredSocketId = 0;
    uint32_t maxEdgeId = 0;

    for (const auto& [nodeKey, savedNode] : state.nodes) {
        if (!savedNode.id.isValid() || nodeKey != savedNode.id.value) {
            errorMessage = "Saved node has an invalid id.";
            return false;
        }

        const NodeTypeDefinition* definition = registry->findNodeType(savedNode.typeId);
        if (!definition) {
            errorMessage = "Saved graph references unknown node type: " + savedNode.typeId;
            return false;
        }

        NodeGraphNode rebuiltNode{};
        rebuiltNode.id = savedNode.id;
        rebuiltNode.typeId = definition->id;
        rebuiltNode.category = definition->category;
        rebuiltNode.title = savedNode.title.empty() ? definition->displayName : savedNode.title;
        rebuiltNode.x = savedNode.x;
        rebuiltNode.y = savedNode.y;
        rebuiltNode.displayEnabled = savedNode.displayEnabled;
        rebuiltNode.frozen = savedNode.frozen;
        rebuiltNode.inputs = candidate.buildSocketsFromInterface(*definition, NodeGraphSocketDirection::Input);
        rebuiltNode.outputs = candidate.buildSocketsFromInterface(*definition, NodeGraphSocketDirection::Output);

        if (!patchSocketIds(rebuiltNode.inputs, savedNode.inputs, errorMessage) ||
            !patchSocketIds(rebuiltNode.outputs, savedNode.outputs, errorMessage)) {
            std::ostringstream ss;
            ss << "Node " << savedNode.id.value << ": " << errorMessage;
            errorMessage = ss.str();
            return false;
        }

        for (const NodeGraphParamDefinition& parameterDefinition : definition->parameters) {
            rebuiltNode.parameters.push_back(makeNodeGraphParamValue(parameterDefinition));
        }

        for (const NodeGraphParamValue& savedParameter : savedNode.parameters) {
            const NodeGraphParamDefinition* parameterDefinition = findNodeParamDefinition(*definition, savedParameter.id);
            if (!parameterDefinition) {
                continue;
            }

            NodeGraphParamValue restoredParameter = savedParameter;
            if (!normalizeNodeGraphParamValue(*parameterDefinition, restoredParameter) ||
                !validateNodeGraphParamValue(*parameterDefinition, restoredParameter)) {
                std::ostringstream ss;
                ss << "Node " << savedNode.id.value << " has an invalid saved parameter value.";
                errorMessage = ss.str();
                return false;
            }

            if (NodeGraphParamValue* existingParameter = findNodeParamValue(rebuiltNode, restoredParameter.id)) {
                *existingParameter = std::move(restoredParameter);
            } else {
                rebuiltNode.parameters.push_back(std::move(restoredParameter));
            }
        }

        maxNodeId = std::max(maxNodeId, rebuiltNode.id.value);
        maxRestoredSocketId = std::max(maxRestoredSocketId, maxSocketId(rebuiltNode));
        candidate.nodes[rebuiltNode.id.value] = std::move(rebuiltNode);
    }

    for (const auto& [edgeKey, savedEdge] : state.edges) {
        if (!savedEdge.id.isValid() || edgeKey != savedEdge.id.value) {
            errorMessage = "Saved edge has an invalid id.";
            return false;
        }

        const NodeGraphNode* fromNode = candidate.findNode(savedEdge.fromNode);
        const NodeGraphNode* toNode = candidate.findNode(savedEdge.toNode);
        if (!fromNode || !toNode) {
            errorMessage = "Saved edge references a missing node.";
            return false;
        }
        if (!findSocket(*fromNode, savedEdge.fromSocket, NodeGraphSocketDirection::Output) ||
            !findSocket(*toNode, savedEdge.toSocket, NodeGraphSocketDirection::Input)) {
            errorMessage = "Saved edge references a missing or invalid socket.";
            return false;
        }

        std::string validationError;
        if (!NodeGraphValidator::canCreateConnection(
                candidate,
                registry->typeRegistry(),
                savedEdge.fromNode,
                savedEdge.fromSocket,
                savedEdge.toNode,
                savedEdge.toSocket,
                validationError)) {
            errorMessage = "Saved edge is invalid: " + validationError;
            return false;
        }

        candidate.edges[savedEdge.id.value] = savedEdge;
        maxEdgeId = std::max(maxEdgeId, savedEdge.id.value);
    }

    candidate.nextNodeId = std::max(serializedNextNodeId, maxNodeId + 1);
    candidate.nextSocketId = std::max(serializedNextSocketId, maxRestoredSocketId + 1);
    candidate.nextEdgeId = std::max(serializedNextEdgeId, maxEdgeId + 1);
    candidate.revision = revision + 1;

    nodes = std::move(candidate.nodes);
    edges = std::move(candidate.edges);
    nextNodeId = candidate.nextNodeId;
    nextSocketId = candidate.nextSocketId;
    nextEdgeId = candidate.nextEdgeId;
    revision = candidate.revision;
    return true;
}

void NodeGraph::getNextIds(uint32_t& outNodeId, uint32_t& outSocketId, uint32_t& outEdgeId) const {
    outNodeId = nextNodeId;
    outSocketId = nextSocketId;
    outEdgeId = nextEdgeId;
}

const NodeGraphNode* NodeGraph::findNode(NodeGraphNodeId nodeId) const {
    auto it = nodes.find(nodeId.value);
    return it != nodes.end() ? &it->second : nullptr;
}

NodeGraphNode* NodeGraph::findNode(NodeGraphNodeId nodeId) {
    auto it = nodes.find(nodeId.value);
    return it != nodes.end() ? &it->second : nullptr;
}

NodeGraphSocketId NodeGraph::allocateSocketId() {
    return NodeGraphSocketId{nextSocketId++};
}

std::vector<NodeGraphSocket> NodeGraph::buildSocketsFromInterface(
    const NodeTypeDefinition& definition,
    NodeGraphSocketDirection direction) {
    std::vector<NodeGraphSocket> sockets;
    for (const NodeSocketSignature& socketSignature : definition.sockets) {
        if (socketSignature.direction != direction) {
            continue;
        }

        sockets.push_back(
            {
                allocateSocketId(),
                socketSignature.name,
                socketSignature.valueType,
                socketSignature.direction,
                socketSignature.contract,
                socketSignature.variadic,
            });
    }

    return sockets;
}

void NodeGraph::bumpRevision() {
    ++revision;
}
