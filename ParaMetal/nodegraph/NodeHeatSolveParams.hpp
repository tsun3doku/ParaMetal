#pragma once

#include "NodeGraphTypes.hpp"
#include "NodeHeatMaterialPresets.hpp"

#include <vector>

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatPreviewSettings {
    bool showHeatOverlay = false;
    bool showFluxVectors = false;
    bool showHeatPalette = false;
    double fluxVectorScale = 1.0;
};

struct HeatSolveNodeParams {
    bool enabled = false;
    bool paused = false;
    bool resetRequested = false;
    double cellSize = 0.005;
    double contactThermalConductance = 16000.0;
    HeatPreviewSettings preview{};
    std::vector<HeatMaterialBindingRow> materialBindingRows;
};

HeatSolveNodeParams readHeatSolveNodeParams(const NodeGraphNode& node);
bool writeHeatSolveNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatSolveNodeParams& params);
std::vector<HeatMaterialBinding> makeHeatPayloadMaterialBindings(const HeatSolveNodeParams& params);