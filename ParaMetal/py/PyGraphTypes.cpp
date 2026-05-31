#include "PyGraphTypes.hpp"

py::object paramValueToPy(const NodeGraphParamValue& value) {
    switch (value.type) {
        case NodeGraphParamType::Float:
            return py::float_(value.floatValue);
        case NodeGraphParamType::Int:
            return py::int_(value.intValue);
        case NodeGraphParamType::Bool:
            return py::bool_(value.boolValue);
        case NodeGraphParamType::String:
            return py::str(value.stringValue);
        case NodeGraphParamType::Enum:
            return py::str(value.enumValue);
        default:
            return py::none();
    }
}

NodeGraphParamValue pyToParamValue(uint32_t paramId, NodeGraphParamType type, py::object value) {
    NodeGraphParamValue result;
    result.id = paramId;
    result.type = type;
    switch (type) {
        case NodeGraphParamType::Float:
            result.floatValue = value.cast<double>();
            break;
        case NodeGraphParamType::Int:
            result.intValue = value.cast<int64_t>();
            break;
        case NodeGraphParamType::Bool:
            result.boolValue = value.cast<bool>();
            break;
        case NodeGraphParamType::String:
            result.stringValue = value.cast<std::string>();
            break;
        case NodeGraphParamType::Enum:
            result.enumValue = value.cast<std::string>();
            break;
        default:
            break;
    }
    return result;
}

PySocket::PySocket(NodeGraphBridge* b, NodeGraphNodeId n, NodeGraphSocketId s)
    : bridge(b), nodeId(n), socketId(s) {}

std::string PySocket::name() const {
    if (!bridge) return "";
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return "";
    const NodeGraphSocket* s = node.input(socketId);
    if (!s) s = node.output(socketId);
    return s ? s->name : "";
}

NodeGraphValueType PySocket::valueType() const {
    if (!bridge) return NodeGraphValueType::None;
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return NodeGraphValueType::None;
    const NodeGraphSocket* s = node.input(socketId);
    if (!s) s = node.output(socketId);
    return s ? s->valueType : NodeGraphValueType::None;
}

PyEdge::PyEdge(NodeGraphBridge* b, NodeGraphEdgeId id) : bridge(b), edgeId(id) {}

PyNode PyEdge::from_node() const {
    NodeGraphEdge e = getEdge();
    return PyNode(bridge, e.fromNode);
}
PySocket PyEdge::from_socket() const {
    NodeGraphEdge e = getEdge();
    return PySocket(bridge, e.fromNode, e.fromSocket);
}
PyNode PyEdge::to_node() const {
    NodeGraphEdge e = getEdge();
    return PyNode(bridge, e.toNode);
}
PySocket PyEdge::to_socket() const {
    NodeGraphEdge e = getEdge();
    return PySocket(bridge, e.toNode, e.toSocket);
}

NodeGraphEdge PyEdge::getEdge() const {
    if (!bridge) return NodeGraphEdge{};
    NodeGraphState state = bridge->state();
    auto it = state.edges.find(edgeId.value);
    return (it != state.edges.end()) ? it->second : NodeGraphEdge{};
}

PyNode::PyNode(NodeGraphBridge* b, NodeGraphNodeId id) : bridge(b), nodeId(id) {}

std::string PyNode::name() const {
    if (!bridge) return "";
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return "";
    return node.title;
}

std::string PyNode::type_id() const {
    if (!bridge) return "";
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return "";
    return node.typeId;
}

py::object PyNode::get(const std::string& paramName) const {
    if (!bridge) return py::none();

    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return py::none();

    const NodeGraphRegistry& registry = bridge->getRegistry();
    const NodeTypeDefinition* typeDef = registry.findNodeType(node.typeId);
    if (!typeDef) return py::none();

    const NodeGraphParamDefinition* paramDef = nullptr;
    for (const auto& p : typeDef->parameters) {
        if (p.name == paramName) {
            paramDef = &p;
            break;
        }
    }
    if (!paramDef) {
        throw py::key_error("Parameter not found: " + paramName);
    }

    for (const auto& pv : node.parameters) {
        if (pv.id == paramDef->id) {
            return paramValueToPy(pv);
        }
    }

    NodeGraphParamValue defaultValue;
    defaultValue.id = paramDef->id;
    defaultValue.type = paramDef->type;
    defaultValue.floatValue = paramDef->defaultFloatValue;
    defaultValue.intValue = paramDef->defaultIntValue;
    defaultValue.boolValue = paramDef->defaultBoolValue;
    defaultValue.stringValue = paramDef->defaultStringValue;
    return paramValueToPy(defaultValue);
}

void PyNode::set(const std::string& paramName, py::object value) {
    if (!bridge) return;

    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return;

    const NodeGraphRegistry& registry = bridge->getRegistry();
    const NodeTypeDefinition* typeDef = registry.findNodeType(node.typeId);
    if (!typeDef) return;

    const NodeGraphParamDefinition* paramDef = nullptr;
    for (const auto& p : typeDef->parameters) {
        if (p.name == paramName) {
            paramDef = &p;
            break;
        }
    }
    if (!paramDef) {
        throw py::key_error("Parameter not found: " + paramName);
    }

    NodeGraphParamValue paramValue = pyToParamValue(paramDef->id, paramDef->type, value);
    bridge->setNodeParameter(nodeId, paramValue);
}

PySocket PyNode::input(const std::string& name) const {
    if (!bridge) throw py::key_error("No bridge available");
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) throw py::key_error("Node not found");
    for (const auto& s : node.inputs) {
        if (s.name == name) {
            return PySocket(bridge, nodeId, s.id);
        }
    }
    throw py::key_error("Input socket not found: " + name);
}

PySocket PyNode::output(const std::string& name) const {
    if (!bridge) throw py::key_error("No bridge available");
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) throw py::key_error("Node not found");
    for (const auto& s : node.outputs) {
        if (s.name == name) {
            return PySocket(bridge, nodeId, s.id);
        }
    }
    throw py::key_error("Output socket not found: " + name);
}

std::vector<PySocket> PyNode::inputs() const {
    std::vector<PySocket> result;
    if (!bridge) return result;
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return result;
    for (const auto& s : node.inputs) {
        result.emplace_back(bridge, nodeId, s.id);
    }
    return result;
}

std::vector<PySocket> PyNode::outputs() const {
    std::vector<PySocket> result;
    if (!bridge) return result;
    NodeGraphNode node{};
    if (!bridge->getNode(nodeId, node)) return result;
    for (const auto& s : node.outputs) {
        result.emplace_back(bridge, nodeId, s.id);
    }
    return result;
}
