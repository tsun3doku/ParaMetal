#pragma once

#include "nodegraph/NodeGraphTypes.hpp"

#include <cstdint>
#include <string>

class NodeGraph;

namespace NodeParamUtils {

bool canEditNode(NodeGraph* graph, NodeGraphNodeId nodeId);
bool loadNode(NodeGraph* graph, NodeGraphNodeId nodeId, NodeGraphNode& outNode);

bool readBoolParam(const NodeGraphNode& node, uint32_t parameterId, bool defaultValue = false);
double readFloatParam(const NodeGraphNode& node, uint32_t parameterId, double defaultValue);
int readIntParam(const NodeGraphNode& node, uint32_t parameterId, int defaultValue);
std::string readStringParam(const NodeGraphNode& node, uint32_t parameterId);
int readEnumParam(const NodeGraphNode& node, uint32_t parameterId, int defaultValue);

bool writeParam(NodeGraph* graph, NodeGraphNodeId nodeId, const NodeGraphParamValue& parameter);
bool writeBoolParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, bool value);
bool writeFloatParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, double value);
bool writeIntParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, int64_t value);
bool writeStringParam(NodeGraph* graph, NodeGraphNodeId nodeId, uint32_t parameterId, const std::string& value);

std::string normalizePresetName(const std::string& value);

} 
