#include "NodeVoronoiParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

VoronoiNodeParams readVoronoiNodeParams(const NodeGraphNode& node) {
    constexpr double sdfSize = 0.005;
    constexpr int voxelResolution = 128;

    VoronoiNodeParams params{};
    params.sdfSize = NodeParamUtils::readFloatParam(node, nodegraphparams::voronoi::SDFSize, sdfSize);
    params.voxelResolution = NodeParamUtils::readIntParam(node, nodegraphparams::voronoi::VoxelResolution, voxelResolution);
    if (params.sdfSize <= 0.0) { params.sdfSize = sdfSize; }
    if (params.voxelResolution <= 0) { params.voxelResolution = voxelResolution; }
    params.preview.showVoronoi = NodeParamUtils::readBoolParam(node, nodegraphparams::voronoi::ShowVoronoi, false);
    params.preview.showPoints = NodeParamUtils::readBoolParam(node, nodegraphparams::voronoi::ShowPoints, false);
    return params;
}

bool writeVoronoiNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const VoronoiNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::SDFSize, NodeGraphParamType::Float, params.sdfSize}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::VoxelResolution, NodeGraphParamType::Int, 0.0, params.voxelResolution}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ShowVoronoi, NodeGraphParamType::Bool, 0.0, 0, params.preview.showVoronoi}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ShowPoints, NodeGraphParamType::Bool, 0.0, 0, params.preview.showPoints});
}
