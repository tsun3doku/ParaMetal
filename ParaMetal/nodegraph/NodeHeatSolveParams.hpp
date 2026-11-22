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

namespace nodegraphparams::heatsolve {
constexpr uint32_t Enabled = 1;
constexpr uint32_t Paused = 2;
constexpr uint32_t ResetRequested = 3;
constexpr uint32_t ShowHeatOverlay = 5;
constexpr uint32_t ShowFluxVectors = 7;
constexpr uint32_t FluxVectorScale = 8;
constexpr uint32_t ShowHeatPalette = 9;
constexpr uint32_t RewindFrame = 10;
constexpr uint32_t SimulationDuration = 11;
constexpr uint32_t ShowDomainBox = 14;
constexpr uint32_t ShowFluidVolume = 15;
constexpr uint32_t FluidVolumeOpacity = 16;
constexpr uint32_t SolverMaxIterations = 30;
constexpr uint32_t SolverConvergenceTolerance = 31;
constexpr uint32_t ShowIsocontours = 32;
constexpr uint32_t IsocontourSpacingC = 34;
}
