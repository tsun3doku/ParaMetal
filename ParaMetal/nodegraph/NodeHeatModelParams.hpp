#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatModelData.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatModelNodeParams {
    HeatMaterialPresetId materialPreset = HeatMaterialPresetId::Custom;
    double density = 1000.0;
    double specificHeat = 1000.0;
    double conductivity = 1.0;
    double initialTemperatureC = HeatSimDefaults::ambientTemperatureC;
    BoundaryCondition::Type boundaryConditionType = BoundaryCondition::Type::Adiabatic;
    double boundaryTemperatureC = HeatSimDefaults::ambientTemperatureC;
    double heatFlux = 0.0;
    double heatTransferCoefficient = 0.0;
    double volumetricPowerDensity = 0.0;
};

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node);
bool writeHeatModelNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatModelNodeParams& params);
