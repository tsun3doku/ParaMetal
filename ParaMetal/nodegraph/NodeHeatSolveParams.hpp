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
    HeatPreviewSettings preview{};

    uint32_t domainBoundaryTypes[6] = {};
    float domainBoundaryTemperaturesC[6] = {
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC
    };
    float domainBoundaryHeatFluxes[6] = {};
    float domainBoundaryHeatTransferCoefficients[6] = {};
};

HeatSolveNodeParams readHeatSolveNodeParams(const NodeGraphNode& node);
bool writeHeatSolveNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatSolveNodeParams& params);
