#include "NodeHeatModelParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node) {
    HeatModelNodeParams params{};
    params.density = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::Density,
        1000.0);
    params.specificHeat = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::SpecificHeat,
        1000.0);
    params.conductivity = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::Conductivity,
        50.0);
    params.initialTemperature = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::InitialTemperature,
        1.0);
    params.boundaryCondition = static_cast<HeatBoundaryCondition>(
        NodePanelUtils::readEnumParam(node, nodegraphparams::heatmodel::BoundaryCondition, 0));
    params.fixedTemperatureValue = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::FixedTemperatureValue,
        1.0);
    return params;
}

bool writeHeatModelNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatModelNodeParams& params) {
    bool ok = true;
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::Density,
            NodeGraphParamType::Float,
            params.density});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::SpecificHeat,
            NodeGraphParamType::Float,
            params.specificHeat});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::Conductivity,
            NodeGraphParamType::Float,
            params.conductivity});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::InitialTemperature,
            NodeGraphParamType::Float,
            params.initialTemperature});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::BoundaryCondition,
            NodeGraphParamType::Enum,
            0.0,
            static_cast<int64_t>(params.boundaryCondition)});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::FixedTemperatureValue,
            NodeGraphParamType::Float,
            params.fixedTemperatureValue});
    return ok;
}
