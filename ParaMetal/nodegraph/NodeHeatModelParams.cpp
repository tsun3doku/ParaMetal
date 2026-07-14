#include "NodeHeatModelParams.hpp"
#include "heat/HeatSystemPresets.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeHeatMaterialPresets.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

HeatModelNodeParams readHeatModelNodeParams(const NodeGraphNode& node) {
    HeatModelNodeParams params{};
    if (const NodeGraphParamValue* presetValue = findNodeParamValue(node, nodegraphparams::heatmodel::MaterialPreset)) {
        std::string presetName;
        if (tryGetParamEnum(*presetValue, presetName)) {
            HeatMaterialPresetId presetId = HeatMaterialPresetId::Custom;
            if (tryResolveHeatPresetId(presetName, presetId)) {
                params.materialPreset = presetId;
            }
        }
    }
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
    params.initialTemperatureC = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::InitialTemperatureC,
        HeatSimDefaults::ambientTemperatureC);
    params.boundaryConditionType = static_cast<BoundaryCondition::Type>(
        NodePanelUtils::readEnumParam(node, nodegraphparams::heatmodel::BoundaryCondition, 0));
    params.boundaryTemperatureC = NodePanelUtils::readFloatParam(
        node,
        nodegraphparams::heatmodel::DirichletTemperatureC,
        HeatSimDefaults::ambientTemperatureC);
    params.heatFlux = NodePanelUtils::readFloatParam(node, nodegraphparams::heatmodel::HeatFlux, 0.0);
    params.heatTransferCoefficient = NodePanelUtils::readFloatParam(
        node, nodegraphparams::heatmodel::HeatTransferCoefficient, 0.0);
    params.volumetricPowerDensity = NodePanelUtils::readFloatParam(
        node, nodegraphparams::heatmodel::VolumetricPowerDensity, 0.0);
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
            nodegraphparams::heatmodel::MaterialPreset,
            NodeGraphParamType::Enum,
            0.0,
            0,
            false,
            "",
            heatMaterialPresetName(params.materialPreset)});
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
            nodegraphparams::heatmodel::InitialTemperatureC,
            NodeGraphParamType::Float,
            params.initialTemperatureC});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::BoundaryCondition,
            NodeGraphParamType::Enum,
            0.0,
            static_cast<int64_t>(params.boundaryConditionType)});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{
            nodegraphparams::heatmodel::DirichletTemperatureC,
            NodeGraphParamType::Float,
            params.boundaryTemperatureC});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{nodegraphparams::heatmodel::HeatFlux, NodeGraphParamType::Float, params.heatFlux});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{nodegraphparams::heatmodel::HeatTransferCoefficient, NodeGraphParamType::Float, params.heatTransferCoefficient});
    ok &= editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{nodegraphparams::heatmodel::VolumetricPowerDensity, NodeGraphParamType::Float, params.volumetricPowerDensity});
    return ok;
}
