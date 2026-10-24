#include "NodeHeatModelParams.hpp"
#include "heat/HeatSystemPresets.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeHeatMaterialPresets.hpp"
#include "nodegraph/NodeParamUtils.hpp"

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
    params.density = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::Density, HeatSimDefaults::density);
    params.specificHeat = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::SpecificHeat, HeatSimDefaults::specificHeat);
    params.conductivity = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::Conductivity, HeatSimDefaults::conductivity);
    params.initialTemperatureC = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::InitialTemperatureC, HeatSimDefaults::ambientTemperatureC);
    params.volumetricPowerDensity = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::VolumetricPowerDensity, 0.0);
    params.prescribedTemperatureEnabled = NodeParamUtils::readBoolParam(node, nodegraphparams::heatmodel::PrescribedTemperatureEnabled, false);
    params.prescribedTemperatureC = NodeParamUtils::readFloatParam(node, nodegraphparams::heatmodel::PrescribedTemperatureC, HeatSimDefaults::ambientTemperatureC);
    return params;
}

bool writeHeatModelNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatModelNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::MaterialPreset, NodeGraphParamType::Enum, 0.0, 0, false, "", heatMaterialPresetName(params.materialPreset)}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::Density, NodeGraphParamType::Float, params.density}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::SpecificHeat, NodeGraphParamType::Float, params.specificHeat}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::Conductivity, NodeGraphParamType::Float, params.conductivity}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::InitialTemperatureC, NodeGraphParamType::Float, params.initialTemperatureC}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::VolumetricPowerDensity, NodeGraphParamType::Float, params.volumetricPowerDensity}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::PrescribedTemperatureEnabled, NodeGraphParamType::Bool, 0.0, 0, params.prescribedTemperatureEnabled}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatmodel::PrescribedTemperatureC, NodeGraphParamType::Float, params.prescribedTemperatureC});
}
