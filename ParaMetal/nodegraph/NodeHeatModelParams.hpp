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
    double volumetricPowerDensity = 0.0;
    bool prescribedTemperatureEnabled = false;
    double prescribedTemperatureC = HeatSimDefaults::ambientTemperatureC;
};

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node);
bool writeHeatModelNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatModelNodeParams& params);

namespace nodegraphparams::heatmodel {
constexpr uint32_t Density = 1;
constexpr uint32_t SpecificHeat = 2;
constexpr uint32_t Conductivity = 3;
constexpr uint32_t InitialTemperatureC = 4;
constexpr uint32_t MaterialPreset = 7;
constexpr uint32_t VolumetricPowerDensity = 10;
constexpr uint32_t PrescribedTemperatureEnabled = 11;
constexpr uint32_t PrescribedTemperatureC = 12;
}
