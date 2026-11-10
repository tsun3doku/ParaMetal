#include "NodeCFDParams.hpp"
#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeParamUtils.hpp"

CFDNodeParams readCFDNodeParams(const NodeGraphNode& node) {
    CFDNodeParams params{};
    params.enabled = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::Enabled, true);
    params.targetCourant = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::TargetCourant, 0.5);
    params.timeStep = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::TimeStep, 1.0 / 60.0);
    params.initialVelocity.x = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityX, 0.0));
    params.initialVelocity.y = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityY, 0.0));
    params.initialVelocity.z = static_cast<float>(NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::VelocityZ, 0.0));
    params.pressureIterations = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::PressureIterations, 200.0);
    params.pressureTolerance = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::PressureTolerance, 1e-4);
    params.simulationDuration = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::Duration, 10.0);
    params.preview.showStreamlines = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::ShowStreamlines, false);
    params.preview.showDomainBox = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::ShowDomainBox, false);
    params.preview.showVelocityPalette = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::ShowVelocityPalette, false);

    return params;
}

bool writeCFDNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const CFDNodeParams& params) {
    const bool baseOk =
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::Enabled, NodeGraphParamType::Bool, 0.0, 0, params.enabled}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::TargetCourant, NodeGraphParamType::Float, params.targetCourant}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::TimeStep, NodeGraphParamType::Float, params.timeStep}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityX, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.x)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityY, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.y)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::VelocityZ, NodeGraphParamType::Float, static_cast<double>(params.initialVelocity.z)}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::PressureIterations, NodeGraphParamType::Int, params.pressureIterations}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::PressureTolerance, NodeGraphParamType::Float, params.pressureTolerance}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::Duration, NodeGraphParamType::Float, params.simulationDuration}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::ShowStreamlines, NodeGraphParamType::Bool, 0.0, 0, params.preview.showStreamlines}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::ShowDomainBox, NodeGraphParamType::Bool, 0.0, 0, params.preview.showDomainBox}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::cfd::ShowVelocityPalette, NodeGraphParamType::Bool, 0.0, 0, params.preview.showVelocityPalette});

    return baseOk;
}
