#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>

class NodeGraphEditor;
struct NodeGraphNode;

struct SerialTemperatureNodeParams {
    bool enabled = true;
    std::string portName;
    uint32_t baudRate = 115200;
};

SerialTemperatureNodeParams readSerialTemperatureNodeParams(const NodeGraphNode& node);
bool writeSerialTemperatureNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId,
    const SerialTemperatureNodeParams& params);
