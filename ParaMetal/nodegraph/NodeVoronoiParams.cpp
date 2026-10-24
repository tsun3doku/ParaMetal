#include "NodeVoronoiParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"
#include <algorithm>

VoronoiNodeParams readVoronoiNodeParams(const NodeGraphNode& node) {
    constexpr double cellSize = 0.005;
    constexpr int voxelResolution = 128;
    constexpr double volumePadding = 0.01;
    constexpr double contactRange = voronoi::DefaultContactRange;

    VoronoiNodeParams params{};
    params.cellSize = NodeParamUtils::readFloatParam(node, nodegraphparams::voronoi::CellSize, cellSize);
    params.voxelResolution = NodeParamUtils::readIntParam(node, nodegraphparams::voronoi::VoxelResolution, voxelResolution);
    params.volumePadding = NodeParamUtils::readFloatParam(node, nodegraphparams::voronoi::VolumePadding, volumePadding);
    params.contactRange = NodeParamUtils::readFloatParam(node, nodegraphparams::voronoi::ContactRange, contactRange);
    if (params.cellSize <= 0.0) { params.cellSize = cellSize; }
    if (params.voxelResolution <= 0) { params.voxelResolution = voxelResolution; }
    if (params.volumePadding < 0.0) { params.volumePadding = 0.0; }
    if (params.contactRange < 0.0) { params.contactRange = contactRange; }
    params.preview.showVoronoi = NodeParamUtils::readBoolParam(node, nodegraphparams::voronoi::ShowVoronoi, false);
    params.preview.showPoints = NodeParamUtils::readBoolParam(node, nodegraphparams::voronoi::ShowPoints, false);
    params.preview.showContactManifold = NodeParamUtils::readBoolParam(node, nodegraphparams::voronoi::ShowContactManifold, false);
    return params;
}

bool writeVoronoiNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const VoronoiNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::CellSize, NodeGraphParamType::Float, params.cellSize}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::VoxelResolution, NodeGraphParamType::Int, 0.0, params.voxelResolution}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::VolumePadding, NodeGraphParamType::Float, params.volumePadding}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ContactRange, NodeGraphParamType::Float, params.contactRange}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ShowVoronoi, NodeGraphParamType::Bool, 0.0, 0, params.preview.showVoronoi}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ShowPoints, NodeGraphParamType::Bool, 0.0, 0, params.preview.showPoints}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::voronoi::ShowContactManifold, NodeGraphParamType::Bool, 0.0, 0, params.preview.showContactManifold});
}
