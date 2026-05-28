#include "NodeHeatSolveParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

HeatSolveNodeParams readHeatSolveNodeParams(const NodeGraphNode& node) {
    HeatSolveNodeParams params{};
    params.enabled = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::Enabled, false);
    params.paused = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::Paused, false);
    params.resetRequested = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::ResetRequested, false);
    params.cellSize = NodePanelUtils::readFloatParam(node, nodegraphparams::heatsolve::CellSize, 0.005);
    params.contactThermalConductance = NodePanelUtils::readFloatParam(node, nodegraphparams::heatsolve::ContactThermalConductance, 16000.0);
    params.preview.showHeatOverlay = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowHeatOverlay, false);
    params.preview.showFluxVectors = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowFluxVectors, false);
    params.preview.showHeatPalette = NodePanelUtils::readBoolParam(node, nodegraphparams::heatsolve::ShowHeatPalette, false);
    params.preview.fluxVectorScale = NodePanelUtils::readFloatParam(node, nodegraphparams::heatsolve::FluxVectorScale, 1.0);

    const NodeGraphParamValue* materialBindingsValue = findNodeParamValue(node, nodegraphparams::heatsolve::MaterialBindings);
    if (materialBindingsValue) {
        params.materialBindingRows = readMaterialBindingRows(*materialBindingsValue);
    }

    return params;
}

bool writeHeatSolveNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatSolveNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::Enabled, NodeGraphParamType::Bool, 0.0, 0, params.enabled}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, params.paused}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ResetRequested, NodeGraphParamType::Bool, 0.0, 0, params.resetRequested}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::CellSize, NodeGraphParamType::Float, params.cellSize}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ContactThermalConductance, NodeGraphParamType::Float, params.contactThermalConductance}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowHeatOverlay, NodeGraphParamType::Bool, 0.0, 0, params.preview.showHeatOverlay}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowFluxVectors, NodeGraphParamType::Bool, 0.0, 0, params.preview.showFluxVectors}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::ShowHeatPalette, NodeGraphParamType::Bool, 0.0, 0, params.preview.showHeatPalette}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::heatsolve::FluxVectorScale, NodeGraphParamType::Float, params.preview.fluxVectorScale}) &&
        editor.updateNodeParameter(
            nodeId,
            nodegraphparams::heatsolve::MaterialBindings,
            [&params](NodeGraphParamValue& parameter) {
                return writeMaterialBindingRows(parameter, params.materialBindingRows);
            });
}

std::vector<HeatMaterialBinding> makeHeatPayloadMaterialBindings(const HeatSolveNodeParams& params) {
    return makeHeatMaterialBindings(params.materialBindingRows);
}