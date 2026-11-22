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

namespace nodegraphparams::serialtemperature {
constexpr uint32_t Enabled = 1;
constexpr uint32_t Port = 2;
constexpr uint32_t BaudRate = 3;
}
