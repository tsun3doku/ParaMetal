#include "NodeHeatSolveParams.hpp"

#include "heat/HeatGpuStructs.hpp"
#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

HeatSolveNodeParams readHeatSolveNodeParams(const NodeGraphNode& node) {
    HeatSolveNodeParams params{};
    params.enabled = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::Enabled, false);
    params.paused = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::Paused, false);
    params.resetCounter = static_cast<uint32_t>(NodeParamUtils::readIntParam(node, nodegraphparams::heatsolve::ResetRequested, 0));
    params.rewindFrame = static_cast<uint32_t>(NodeParamUtils::readIntParam(node, nodegraphparams::heatsolve::RewindFrame, heat::NoRewindFrame));
    params.simulationDuration = NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::SimulationDuration, 10.0);
    params.solverMaxIterations = static_cast<uint32_t>(NodeParamUtils::readIntParam(node, nodegraphparams::heatsolve::SolverMaxIterations, 100));
    params.solverConvergenceTolerance = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::SolverConvergenceTolerance, 3e-3));
    params.preview.showHeatOverlay = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowHeatOverlay, false);
    params.preview.showFluxVectors = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowFluxVectors, false);
    params.preview.showHeatPalette = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowHeatPalette, false);
    params.preview.showDomainBox = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowDomainBox, false);
    params.preview.showFluidVolume = NodeParamUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowFluidVolume, false);
    params.preview.fluidVolumeOpacity = NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::FluidVolumeOpacity, 0.25);
    params.preview.fluxVectorScale = NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::FluxVectorScale, 1.0);

    for (uint32_t face = 0; face < 6; ++face) {
        params.domainBoundaryTypes[face] = static_cast<uint32_t>(NodeParamUtils::readEnumParam(node, nodegraphparams::heatsolve::domainBoundaryTypeParam(face), 0));
        params.domainBoundaryTemperaturesC[face] = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::domainBoundaryTempParam(face), HeatSimDefaults::ambientTemperatureC));
        params.domainBoundaryHeatFluxes[face] = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::domainBoundaryFluxParam(face), 0.0));
        params.domainBoundaryHeatTransferCoefficients[face] = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::heatsolve::domainBoundaryHtcParam(face), 0.0));
    }

    return params;
}

bool writeHeatSolveNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatSolveNodeParams& params) {
    const bool baseOk =
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::Enabled, NodeGraphParamType::Bool, 0.0, 0, params.enabled}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, params.paused}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ResetRequested, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(params.resetCounter), false}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::RewindFrame, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(params.rewindFrame), false}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::SimulationDuration, NodeGraphParamType::Float, params.simulationDuration}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::SolverMaxIterations, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(params.solverMaxIterations), false}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::SolverConvergenceTolerance, NodeGraphParamType::Float, static_cast<double>(params.solverConvergenceTolerance)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowHeatOverlay, NodeGraphParamType::Bool, 0.0, 0, params.preview.showHeatOverlay}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowFluxVectors, NodeGraphParamType::Bool, 0.0, 0, params.preview.showFluxVectors}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowHeatPalette, NodeGraphParamType::Bool, 0.0, 0, params.preview.showHeatPalette}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowDomainBox, NodeGraphParamType::Bool, 0.0, 0, params.preview.showDomainBox}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowFluidVolume, NodeGraphParamType::Bool, 0.0, 0, params.preview.showFluidVolume}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::FluidVolumeOpacity, NodeGraphParamType::Float, params.preview.fluidVolumeOpacity}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::FluxVectorScale, NodeGraphParamType::Float, params.preview.fluxVectorScale});

    if (!baseOk) {
        return false;
    }

    for (uint32_t face = 0; face < 6; ++face) {
        const bool faceOk =
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::domainBoundaryTypeParam(face), NodeGraphParamType::Enum, 0.0, static_cast<int64_t>(params.domainBoundaryTypes[face]), false}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::domainBoundaryTempParam(face), NodeGraphParamType::Float, static_cast<double>(params.domainBoundaryTemperaturesC[face])}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::domainBoundaryFluxParam(face), NodeGraphParamType::Float, static_cast<double>(params.domainBoundaryHeatFluxes[face])}) &&
            editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::domainBoundaryHtcParam(face), NodeGraphParamType::Float, static_cast<double>(params.domainBoundaryHeatTransferCoefficients[face])});
        if (!faceOk) {
            return false;
        }
    }

    return true;
}
