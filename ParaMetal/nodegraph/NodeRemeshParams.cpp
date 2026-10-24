#include "NodeRemeshParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

RemeshNodeParams readRemeshNodeParams(const NodeGraphNode& node) {
    RemeshNodeParams params{};
    params.iterations = NodeParamUtils::readIntParam(node, nodegraphparams::remesh::Iterations, params.iterations);
    params.minAngleDegrees = NodeParamUtils::readFloatParam(node, nodegraphparams::remesh::MinAngleDegrees, params.minAngleDegrees);
    params.maxEdgeLength = NodeParamUtils::readFloatParam(node, nodegraphparams::remesh::MaxEdgeLength, params.maxEdgeLength);
    params.stepSize = NodeParamUtils::readFloatParam(node, nodegraphparams::remesh::StepSize, params.stepSize);
    params.preview.showRemeshOverlay = NodeParamUtils::readBoolParam(node, nodegraphparams::remesh::ShowRemeshOverlay, params.preview.showRemeshOverlay);
    params.preview.showFaceNormals = NodeParamUtils::readBoolParam(node, nodegraphparams::remesh::ShowFaceNormals, params.preview.showFaceNormals);
    params.preview.showVertexNormals = NodeParamUtils::readBoolParam(node, nodegraphparams::remesh::ShowVertexNormals, params.preview.showVertexNormals);
    params.normalLength = NodeParamUtils::readFloatParam(node, nodegraphparams::remesh::NormalLength, params.normalLength);
    return params;
}

bool writeRemeshNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const RemeshNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::Iterations, NodeGraphParamType::Int, 0.0, params.iterations}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::MinAngleDegrees, NodeGraphParamType::Float, params.minAngleDegrees}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::MaxEdgeLength, NodeGraphParamType::Float, params.maxEdgeLength}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::StepSize, NodeGraphParamType::Float, params.stepSize}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::ShowRemeshOverlay, NodeGraphParamType::Bool, 0.0, 0, params.preview.showRemeshOverlay}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::ShowFaceNormals, NodeGraphParamType::Bool, 0.0, 0, params.preview.showFaceNormals}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::ShowVertexNormals, NodeGraphParamType::Bool, 0.0, 0, params.preview.showVertexNormals}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::remesh::NormalLength, NodeGraphParamType::Float, params.normalLength});
}
