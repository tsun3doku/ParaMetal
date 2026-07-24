#include "NodeTransformParams.hpp"

#include "NodeGraph.hpp"
#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

TransformNodeParams readTransformNodeParams(const NodeGraphNode& node) {
    TransformNodeParams params{};
    params.translateX = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::TranslateX, 0.0);
    params.translateY = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::TranslateY, 0.0);
    params.translateZ = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::TranslateZ, 0.0);
    params.rotateXDegrees = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::RotateXDegrees, 0.0);
    params.rotateYDegrees = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::RotateYDegrees, 0.0);
    params.rotateZDegrees = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::RotateZDegrees, 0.0);
    params.scaleX = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::ScaleX, 1.0);
    params.scaleY = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::ScaleY, 1.0);
    params.scaleZ = NodeParamUtils::readFloatParam(node, nodegraphparams::transform::ScaleZ, 1.0);
    return params;
}

bool writeTransformNodeParams(NodeGraph& bridge, NodeGraphNodeId nodeId, const TransformNodeParams& params) {
    return bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateX, NodeGraphParamType::Float, params.translateX}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateY, NodeGraphParamType::Float, params.translateY}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateZ, NodeGraphParamType::Float, params.translateZ}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateXDegrees, NodeGraphParamType::Float, params.rotateXDegrees}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateYDegrees, NodeGraphParamType::Float, params.rotateYDegrees}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateZDegrees, NodeGraphParamType::Float, params.rotateZDegrees}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleX, NodeGraphParamType::Float, params.scaleX}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleY, NodeGraphParamType::Float, params.scaleY}) &&
        bridge.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleZ, NodeGraphParamType::Float, params.scaleZ});
}

bool writeTransformNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const TransformNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateX, NodeGraphParamType::Float, params.translateX}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateY, NodeGraphParamType::Float, params.translateY}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::TranslateZ, NodeGraphParamType::Float, params.translateZ}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateXDegrees, NodeGraphParamType::Float, params.rotateXDegrees}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateYDegrees, NodeGraphParamType::Float, params.rotateYDegrees}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::RotateZDegrees, NodeGraphParamType::Float, params.rotateZDegrees}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleX, NodeGraphParamType::Float, params.scaleX}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleY, NodeGraphParamType::Float, params.scaleY}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::transform::ScaleZ, NodeGraphParamType::Float, params.scaleZ});
}
