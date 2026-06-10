#include "NodePointsParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

PointsNodeParams readPointsNodeParams(const NodeGraphNode& node) {
    PointsNodeParams params{};

    const NodeGraphParamValue* countParam = findNodeParamValue(node, nodegraphparams::points::PointCount);
    if (countParam && countParam->type == NodeGraphParamType::Int) {
        params.pointCount = static_cast<uint32_t>(countParam->intValue);
    }

    const NodeGraphParamValue* dimXParam = findNodeParamValue(node, nodegraphparams::points::DimX);
    if (dimXParam && dimXParam->type == NodeGraphParamType::Float) {
        params.dimX = static_cast<float>(dimXParam->floatValue);
    }

    const NodeGraphParamValue* dimYParam = findNodeParamValue(node, nodegraphparams::points::DimY);
    if (dimYParam && dimYParam->type == NodeGraphParamType::Float) {
        params.dimY = static_cast<float>(dimYParam->floatValue);
    }

    const NodeGraphParamValue* dimZParam = findNodeParamValue(node, nodegraphparams::points::DimZ);
    if (dimZParam && dimZParam->type == NodeGraphParamType::Float) {
        params.dimZ = static_cast<float>(dimZParam->floatValue);
    }

    return params;
}

bool writePointsNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const PointsNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::points::PointCount, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(params.pointCount)}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::points::DimX, NodeGraphParamType::Float, static_cast<double>(params.dimX)}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::points::DimY, NodeGraphParamType::Float, static_cast<double>(params.dimY)}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::points::DimZ, NodeGraphParamType::Float, static_cast<double>(params.dimZ)});
}
