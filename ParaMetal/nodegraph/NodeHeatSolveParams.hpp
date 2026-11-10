#pragma once

#include "NodeGraphTypes.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatSystemPresets.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatPreviewSettings {
    bool showHeatOverlay = false;
    bool showFluxVectors = false;
    bool showHeatPalette = false;
    bool showIsocontours = false;
    bool showDomainBox = false;
    bool showFluidVolume = false;
    double fluidVolumeOpacity = 0.25;
    double fluxVectorScale = 1.0;
};

struct HeatSolveNodeParams {
    bool enabled = false;
    bool paused = false;
    uint32_t resetCounter = 0;
    uint32_t rewindFrame = heat::NoRewindFrame;
    double simulationDuration = 10.0;
    uint32_t solverMaxIterations = 100;        // 0 = keep heat solver default
    float solverConvergenceTolerance = 3e-3f;  // 0 = keep heat solver default
    float isocontourSpacingC = 10.0f;
    HeatPreviewSettings preview{};
};

HeatSolveNodeParams readHeatSolveNodeParams(const NodeGraphNode& node);
bool writeHeatSolveNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatSolveNodeParams& params);
