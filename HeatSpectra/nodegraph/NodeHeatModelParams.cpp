#include "NodeHeatModelParams.hpp"
#include "heat/HeatSystemPresets.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node) {
    HeatModelNodeParams params{};
    params.density = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::Density,
        HeatSimDefaults::density);
    params.specificHeat = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::SpecificHeat,
        HeatSimDefaults::specificHeat);
    params.conductivity = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::Conductivity,
        HeatSimDefaults::conductivity);
    params.initialTemperature = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::InitialTemperature,
        HeatSimDefaults::ambientTemperature);
    params.boundaryCondition = static_cast<HeatBoundaryCondition>(
        NodePanelUtils::readEnumParam(node, nodegraphparams::heatmodel::BoundaryCondition, 0));
    params.fixedTemperatureValue = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::FixedTemperatureValue,
        HeatSimDefaults::ambientTemperature);
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
