#pragma once

#include "NodeGraphDataTypes.hpp"
#include <string>

struct NodeGraphKernelExecutionState;
class NodeGraphRegistry;
class NodeGraph;
struct NodeGraphState;

bool findFirstUpstreamNodeByType(const NodeGraphState& state, NodeGraphNodeId startNodeId, const NodeTypeId& targetTypeId, NodeGraphNodeId& outNodeId);
bool findFirstUpstreamNodeByType(const NodeGraphState& state, uint64_t outputSocketKey, const NodeTypeId& targetTypeId, NodeGraphNodeId& outNodeId);

NodeGraphValueType socketType(const NodeGraphState& state, NodeGraphNodeId nodeId, NodeGraphSocketId socketId);
NodeGraphValueType socketType(const NodeGraph& document, NodeGraphNodeId nodeId, NodeGraphSocketId socketId);

const EvaluatedSocketValue* readEvaluatedInput(
    const NodeGraphNode& node,
    NodeGraphSocketId inputSocketId,
    const NodeGraphKernelExecutionState& executionState);
std::vector<const EvaluatedSocketValue*> readEvaluatedInputs(
    const NodeGraphNode& node,
    NodeGraphSocketId inputSocketId,
    const NodeGraphKernelExecutionState& executionState);
const NodeDataBlock* readInputValue(const EvaluatedSocketValue* input);

NodeGraphParamValue makeNodeGraphParamValue(const NodeGraphParamDefinition& definition);
const NodeGraphParamValue* findNodeParamValue(const NodeGraphNode& node, uint32_t paramId);
NodeGraphParamValue* findNodeParamValue(NodeGraphNode& node, uint32_t paramId);
bool normalizeNodeGraphParamValue(const NodeGraphParamDefinition& definition, NodeGraphParamValue& value);
bool validateNodeGraphParamValue(const NodeGraphParamDefinition& definition, const NodeGraphParamValue& value);

std::string displayNodeLabel(const NodeGraphNode& node);

std::string valueTypeToString(NodeGraphValueType value);

NodeTypeId getNodeTypeId(const NodeTypeId& requestedTypeId);
const NodeGraphParamDefinition* findNodeParamDefinition(const NodeTypeDefinition& definition, uint32_t paramId);

bool tryGetNodeParamFloat(const NodeGraphNode& node, uint32_t paramId, double& outValue);
bool tryGetNodeParamInt(const NodeGraphNode& node, uint32_t paramId, int64_t& outValue);
bool tryGetNodeParamBool(const NodeGraphNode& node, uint32_t paramId, bool& outValue);
bool tryGetNodeParamString(const NodeGraphNode& node, uint32_t paramId, std::string& outValue);
