#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatModelData.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatModelNodeParams {
    double density = 1000.0;
    double specificHeat = 1000.0;
    double conductivity = 1.0;
    double initialTemperature = 293.15;
    HeatBoundaryCondition boundaryCondition = HeatBoundaryCondition::None;
    double fixedTemperatureValue = 293.15;
};

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node);
bool writeHeatModelNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatModelNodeParams& params);
