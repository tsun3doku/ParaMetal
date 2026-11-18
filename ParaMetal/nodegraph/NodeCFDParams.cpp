#include "NodeCFDParams.hpp"
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
    params.preview.showSlice = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::ShowSlice, false);
    params.preview.sliceAxis = NodeParamUtils::readEnumParam(node, nodegraphparams::cfd::SliceAxis, 0);
    params.preview.slicePosition = NodeParamUtils::readFloatParam(node, nodegraphparams::cfd::SlicePosition, 0.5);
    params.preview.showSliceWireframe = NodeParamUtils::readBoolParam(node, nodegraphparams::cfd::ShowSliceWireframe, false);

    return params;
}
