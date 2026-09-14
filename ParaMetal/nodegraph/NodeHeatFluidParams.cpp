#include "NodeHeatFluidParams.hpp"
#include "heat/HeatSystemPresets.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodeParamUtils.hpp"

HeatFluidNodeParams readHeatFluidNodeParams(const NodeGraphNode& node) {
    HeatFluidNodeParams params{};
    params.density = NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::Density,
        1.225);
    params.specificHeat = NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::SpecificHeat,
        1005.0);
    params.conductivity = NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::Conductivity,
        0.026);
    params.initialTemperatureC = NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::InitialTemperatureC,
        HeatSimDefaults::ambientTemperatureC);
    params.inletTemperatureC = NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::InletTemperatureC,
        HeatSimDefaults::ambientTemperatureC);
    params.velocity.x = static_cast<float>(NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::VelocityX,
        0.0));
    params.velocity.y = static_cast<float>(NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::VelocityY,
        0.0));
    params.velocity.z = static_cast<float>(NodeParamUtils::readFloatParam(
        node,
        nodegraphparams::heatfluid::VelocityZ,
        0.0));
    return params;
}

bool writeHeatFluidNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatFluidNodeParams& params) {
    bool ok = true;
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::Density,
            NodeGraphParamType::Float,
            params.density});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::SpecificHeat,
            NodeGraphParamType::Float,
            params.specificHeat});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::Conductivity,
            NodeGraphParamType::Float,
            params.conductivity});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::InitialTemperatureC,
            NodeGraphParamType::Float,
            params.initialTemperatureC});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::InletTemperatureC,
            NodeGraphParamType::Float,
            params.inletTemperatureC});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::VelocityX,
            NodeGraphParamType::Float,
            static_cast<double>(params.velocity.x)});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::VelocityY,
            NodeGraphParamType::Float,
            static_cast<double>(params.velocity.y)});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatfluid::VelocityZ,
            NodeGraphParamType::Float,
            static_cast<double>(params.velocity.z)});
    return ok;
}
