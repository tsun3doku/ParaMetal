#include "NodeSerialTemperatureParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

SerialTemperatureNodeParams readSerialTemperatureNodeParams(const NodeGraphNode& node) {
    SerialTemperatureNodeParams params{};
    params.enabled = NodePanelUtils::readBoolParam(node, nodegraphparams::serialtemperature::Enabled, true);
    params.portName = NodePanelUtils::readStringParam(node, nodegraphparams::serialtemperature::Port);
    const int baudRate = NodePanelUtils::readIntParam(
        node, nodegraphparams::serialtemperature::BaudRate, 115200);
    if (baudRate > 0) params.baudRate = static_cast<uint32_t>(baudRate);
    return params;
}

bool writeSerialTemperatureNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId,
    const SerialTemperatureNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{
               nodegraphparams::serialtemperature::Enabled, NodeGraphParamType::Bool, 0.0, 0, params.enabled}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{
               nodegraphparams::serialtemperature::Port, NodeGraphParamType::String, 0.0, 0, false, params.portName}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{
               nodegraphparams::serialtemperature::BaudRate, NodeGraphParamType::Int, 0.0,
               static_cast<int64_t>(params.baudRate)});
}
