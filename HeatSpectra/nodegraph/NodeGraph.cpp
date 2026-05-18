#include "NodeGraph.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include <cmath>
#include <utility>

NodeGraph::NodeGraph(const NodeGraphRegistry* reg) : registry(reg) {}

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
