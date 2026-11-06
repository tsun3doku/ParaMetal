#include "NodeParamUtils.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphEditor.hpp"

#include <cctype>

namespace NodeParamUtils {

bool canEditNode(NodeGraph* graph, NodeGraphNodeId nodeId) {
    return graph && nodeId.isValid();
}

bool loadNode(NodeGraph* graph, NodeGraphNodeId nodeId, NodeGraphNode& outNode) {
    if (!canEditNode(graph, nodeId)) {
        return false;
    }
    return graph->getNode(nodeId, outNode);
}

bool readBoolParam(const NodeGraphNode& node, uint32_t parameterId, bool defaultValue) {
    bool value = defaultValue;
    if (tryGetNodeParamBool(node, parameterId, value)) {
        return value;
    }
    return defaultValue;
}

double readFloatParam(const NodeGraphNode& node, uint32_t parameterId, double defaultValue) {
    double value = defaultValue;
    if (tryGetNodeParamFloat(node, parameterId, value)) {
        return value;
    }
    return defaultValue;
}

int readIntParam(const NodeGraphNode& node, uint32_t parameterId, int defaultValue) {
    int64_t value = defaultValue;
    if (tryGetNodeParamInt(node, parameterId, value)) {
        return static_cast<int>(value);
    }
    return defaultValue;
}

std::string readStringParam(const NodeGraphNode& node, uint32_t parameterId) {
    std::string value;
    if (tryGetNodeParamString(node, parameterId, value)) {
        return value;
    }
    return {};
}

int readEnumParam(const NodeGraphNode& node, uint32_t parameterId, int defaultValue) {
    const NodeGraphParamValue* param = findNodeParamValue(node, parameterId);
    if (!param) {
        return defaultValue;
    }
    return static_cast<int>(param->intValue);
}

bool writeParam(NodeGraph* graph, NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter) {
    if (!graph) {
        return false;
    }

    NodeGraphEditor editor(*graph);
    return editor.setNodeParameter(nodeId, parameter);
}

bool writeBoolParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, bool value) {
    if (!graph) {
        return false;
    }

    NodeGraphParamValue parameter{};
    parameter.id = parameterId;
    parameter.type = NodeGraphParamType::Bool;
    parameter.boolValue = value;
    return writeParam(graph, nodeId, parameter);
}

bool writeFloatParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, double value) {
    if (!graph) {
        return false;
    }

    NodeGraphParamValue parameter{};
    parameter.id = parameterId;
    parameter.type = NodeGraphParamType::Float;
    parameter.floatValue = value;
    return writeParam(graph, nodeId, parameter);
}

bool writeIntParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, int64_t value) {
    if (!graph) {
        return false;
    }

    NodeGraphParamValue parameter{};
    parameter.id = parameterId;
    parameter.type = NodeGraphParamType::Int;
    parameter.intValue = value;
    return writeParam(graph, nodeId, parameter);
}

bool writeStringParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, const std::string& value) {
    if (!graph) {
        return false;
    }

    NodeGraphParamValue parameter{};
    parameter.id = parameterId;
    parameter.type = NodeGraphParamType::String;
    parameter.stringValue = value;
    return writeParam(graph, nodeId, parameter);
}

std::string normalizePresetName(const std::string& value) {
    std::string normalized;
    normalized.reserve(value.size());
    for (char character : value) {
        const unsigned char u = static_cast<unsigned char>(character);
        if (std::isalnum(u) == 0) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(u)));
    }
    return normalized;
}

} 
