#include "NodeGraphUtils.hpp"
#include "NodeGraphKernels.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphTypes.hpp"

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <vector>
#include <sstream>

bool findFirstUpstreamNodeByType(const NodeGraphState& state, NodeGraphNodeId startNodeId, const NodeTypeId& targetTypeId, NodeGraphNodeId& outNodeId) {
    outNodeId = {};
    if (!startNodeId.isValid()) {
        return false;
    }

    std::vector<NodeGraphNodeId> stack{startNodeId};
    std::unordered_set<uint32_t> visitedNodeIds;
    visitedNodeIds.insert(startNodeId.value);

    while (!stack.empty()) {
        const NodeGraphNodeId currentNodeId = stack.back();
        stack.pop_back();

        const NodeGraphNode* currentNode = state.node(currentNodeId);
        if (!currentNode) {
            continue;
        }

        for (const NodeGraphSocket& inputSocket : currentNode->inputs) {
            if (inputSocket.valueType != NodeGraphValueType::Mesh) {
                continue;
            }

            const NodeGraphEdge* incomingEdge = state.incomingEdge(currentNodeId, inputSocket.id);
            if (!incomingEdge || !incomingEdge->fromNode.isValid()) {
                continue;
            }
            if (!visitedNodeIds.insert(incomingEdge->fromNode.value).second) {
                continue;
            }

            const NodeGraphNode* upstreamNode = state.node(incomingEdge->fromNode);
            if (!upstreamNode) {
                continue;
            }
            if (getNodeTypeId(upstreamNode->typeId) == targetTypeId) {
                outNodeId = upstreamNode->id;
                return true;
            }

            stack.push_back(upstreamNode->id);
        }
    }

    return false;
}

bool findFirstUpstreamNodeByType(const NodeGraphState& state, uint64_t outputSocketKey, const NodeTypeId& targetTypeId, NodeGraphNodeId& outNodeId) {
    outNodeId = {};

    NodeGraphNodeId producerNodeId{};
    NodeGraphSocketId producerSocketId{};
    const NodeSocketKey key(outputSocketKey);
    producerNodeId = key.node();
    producerSocketId = key.socket();
    if (!producerNodeId.isValid() || !producerSocketId.isValid()) {
        return false;
    }

    const NodeGraphNode* producerNode = state.node(producerNodeId);
    if (!producerNode) {
        return false;
    }
    if (getNodeTypeId(producerNode->typeId) == targetTypeId) {
        outNodeId = producerNode->id;
        return true;
    }

    return findFirstUpstreamNodeByType(state, producerNodeId, targetTypeId, outNodeId);
}

bool validateNodeGraphParamValue(const NodeGraphParamDefinition& definition, const NodeGraphParamValue& value);

namespace {

bool validateNodeGraphParamFieldValues(
    const std::vector<NodeGraphParamField>& fields,
    const std::vector<NodeGraphParamFieldValue>& fieldValues) {
    if (fields.size() != fieldValues.size()) {
        return false;
    }

    std::unordered_set<std::string> seenFieldNames;
    for (const NodeGraphParamField& field : fields) {
        if (!field.definition || field.name.empty()) {
            return false;
        }

        const auto fieldValueIt = std::find_if(
            fieldValues.begin(),
            fieldValues.end(),
            [&field](const NodeGraphParamFieldValue& fieldValue) {
                return fieldValue.name == field.name;
            });
        if (fieldValueIt == fieldValues.end() || !fieldValueIt->value) {
            return false;
        }
        if (!seenFieldNames.insert(field.name).second) {
            return false;
        }
        if (!validateNodeGraphParamValue(*field.definition, *fieldValueIt->value)) {
            return false;
        }
    }

    return true;
}

} // namespace

const EvaluatedSocketValue* readEvaluatedInput(
    const NodeGraphNode& node,
    NodeGraphSocketId inputSocketId,
    const NodeGraphKernelExecutionState& executionState) {
    const auto edgeIt = executionState.incomingEdgeByInputSocket.find(NodeSocketKey(node.id, inputSocketId).value);
    if (edgeIt == executionState.incomingEdgeByInputSocket.end() || !edgeIt->second) {
        return nullptr;
    }

    const NodeGraphEdge& edge = *edgeIt->second;
    const auto valueIt = executionState.outputBySocket.find(NodeSocketKey(edge.fromNode, edge.fromSocket).value);
    if (valueIt == executionState.outputBySocket.end()) {
        return nullptr;
    }

    return &valueIt->second;
}

std::vector<const EvaluatedSocketValue*> readEvaluatedInputs(
    const NodeGraphNode& node,
    NodeGraphSocketId inputSocketId,
    const NodeGraphKernelExecutionState& executionState) {
    std::vector<const EvaluatedSocketValue*> results;
    const auto edgesIt = executionState.incomingEdgesByInputSocket.find(NodeSocketKey(node.id, inputSocketId).value);
    if (edgesIt == executionState.incomingEdgesByInputSocket.end() || edgesIt->second.empty()) {
        return results;
    }

    results.reserve(edgesIt->second.size());
    for (const NodeGraphEdge* edge : edgesIt->second) {
        if (!edge) continue;
        const auto valueIt = executionState.outputBySocket.find(NodeSocketKey(edge->fromNode, edge->fromSocket).value);
        if (valueIt != executionState.outputBySocket.end()) {
            results.push_back(&valueIt->second);
        }
    }

    return results;
}

const NodeDataBlock* readInputValue(const EvaluatedSocketValue* input) {
    if (!input || input->status != EvaluatedSocketStatus::Value) {
        return nullptr;
    }

    return &input->data;
}

NodeTypeId getNodeTypeId(const NodeTypeId& requestedTypeId) {
    return requestedTypeId.empty() ? nodegraphtypes::Custom : requestedTypeId;
}

std::string displayNodeLabel(const NodeGraphNode& node) {
    if (!node.title.empty()) {
        return node.title;
    }

    if (!node.typeId.empty()) {
        return node.typeId;
    }

    std::ostringstream ss;
    ss << "node#" << node.id.value;
    return ss.str();
}

const NodeGraphParamDefinition* findNodeParamDefinition(const NodeTypeDefinition& definition, uint32_t paramId) {
    for (const NodeGraphParamDefinition& parameter : definition.parameters) {
        if (parameter.id == paramId) {
            return &parameter;
        }
    }
    return nullptr;
}

NodeGraphParamValue makeNodeGraphParamValue(const NodeGraphParamDefinition& definition) {
    NodeGraphParamValue value{};
    value.id = definition.id;
    value.type = definition.type;
    value.floatValue = definition.defaultFloatValue;
    value.intValue = definition.defaultIntValue;
    value.boolValue = definition.defaultBoolValue;
    value.stringValue = definition.defaultStringValue;
    value.enumValue = definition.defaultStringValue;
    if (value.type == NodeGraphParamType::Enum && value.enumValue.empty() && !definition.enumOptions.empty()) {
        value.enumValue = definition.enumOptions.front();
    }
    normalizeNodeGraphParamValue(definition, value);
    if (value.type == NodeGraphParamType::Struct) {
        for (const NodeGraphParamField& field : definition.fields) {
            if (!field.definition) {
                continue;
            }
            value.fieldValues.push_back(makeParamFieldValue(field.name.c_str(), makeNodeGraphParamValue(*field.definition)));
        }
    }
    return value;
}

const NodeGraphParamValue* findNodeParamValue(const NodeGraphNode& node, uint32_t paramId) {
    for (const NodeGraphParamValue& parameter : node.parameters) {
        if (parameter.id == paramId) {
            return &parameter;
        }
    }
    return nullptr;
}

NodeGraphParamValue* findNodeParamValue(NodeGraphNode& node, uint32_t paramId) {
    for (NodeGraphParamValue& parameter : node.parameters) {
        if (parameter.id == paramId) {
            return &parameter;
        }
    }
    return nullptr;
}

bool normalizeNodeGraphParamValue(const NodeGraphParamDefinition& definition, NodeGraphParamValue& value) {
    if (definition.type != value.type) {
        return false;
    }

    if (definition.type != NodeGraphParamType::Enum) {
        return true;
    }

    if (definition.enumOptions.empty()) {
        return false;
    }

    if (!value.enumValue.empty()) {
        const auto optionIt = std::find(
            definition.enumOptions.begin(),
            definition.enumOptions.end(),
            value.enumValue);
        if (optionIt == definition.enumOptions.end()) {
            return false;
        }

        value.intValue = std::distance(definition.enumOptions.begin(), optionIt);
        return true;
    }

    if (value.intValue < 0 || value.intValue >= static_cast<int64_t>(definition.enumOptions.size())) {
        return false;
    }

    value.enumValue = definition.enumOptions[static_cast<std::size_t>(value.intValue)];
    return true;
}

bool validateNodeGraphParamValue(const NodeGraphParamDefinition& definition, const NodeGraphParamValue& value) {
    if (definition.type != value.type) {
        return false;
    }

    switch (definition.type) {
    case NodeGraphParamType::Float:
    case NodeGraphParamType::Int:
    case NodeGraphParamType::Bool:
    case NodeGraphParamType::String:
        return true;
    case NodeGraphParamType::Enum:
        return !value.enumValue.empty()
            && std::find(definition.enumOptions.begin(), definition.enumOptions.end(), value.enumValue)
                != definition.enumOptions.end();
    case NodeGraphParamType::Struct:
        return validateNodeGraphParamFieldValues(definition.fields, value.fieldValues);
    case NodeGraphParamType::Array:
        if (!definition.elementDefinition) {
            return false;
        }
        return std::all_of(
            value.arrayValues.begin(),
            value.arrayValues.end(),
            [&definition](const NodeGraphParamValue& elementValue) {
                return validateNodeGraphParamValue(*definition.elementDefinition, elementValue);
            });
    }

    return false;
}

bool tryGetNodeParamFloat(const NodeGraphNode& node, uint32_t paramId, double& outValue) {
    const NodeGraphParamValue* parameter = findNodeParamValue(node, paramId);
    if (!parameter || parameter->type != NodeGraphParamType::Float) {
        return false;
    }
    outValue = parameter->floatValue;
    return true;
}

bool tryGetNodeParamInt(const NodeGraphNode& node, uint32_t paramId, int64_t& outValue) {
    const NodeGraphParamValue* parameter = findNodeParamValue(node, paramId);
    if (!parameter || parameter->type != NodeGraphParamType::Int) {
        return false;
    }
    outValue = parameter->intValue;
    return true;
}

bool tryGetNodeParamBool(const NodeGraphNode& node, uint32_t paramId, bool& outValue) {
    const NodeGraphParamValue* parameter = findNodeParamValue(node, paramId);
    if (!parameter || parameter->type != NodeGraphParamType::Bool) {
        return false;
    }
    outValue = parameter->boolValue;
    return true;
}

bool tryGetNodeParamString(const NodeGraphNode& node, uint32_t paramId, std::string& outValue) {
    const NodeGraphParamValue* parameter = findNodeParamValue(node, paramId);
    if (!parameter || parameter->type != NodeGraphParamType::String) {
        return false;
    }
    outValue = parameter->stringValue;
    return true;
}

