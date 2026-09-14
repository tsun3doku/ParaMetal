#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatFluidData.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatFluidNodeParams {
    double density = 1.225;
    double specificHeat = 1005.0;
    double conductivity = 0.026;
    double initialTemperatureC = HeatSimDefaults::ambientTemperatureC;
    double inletTemperatureC = HeatSimDefaults::ambientTemperatureC;
    glm::vec3 velocity{0.0f};
};

HeatFluidNodeParams readHeatFluidNodeParams(const NodeGraphNode& node);
bool writeHeatFluidNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatFluidNodeParams& params);
