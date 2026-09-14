#include "NodeMergeParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

MergeNodeParams readMergeNodeParams(const NodeGraphNode& node) {
    MergeNodeParams params{};
    const NodeGraphParamValue* distanceParam = findNodeParamValue(node, nodegraphparams::merge::MergeDistance);
    if (distanceParam && distanceParam->type == NodeGraphParamType::Float) {
        params.mergeDistance = static_cast<float>(distanceParam->floatValue);
    }
    return params;
}

bool writeMergeNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const MergeNodeParams& params) {
    return editor.setNodeParameter(
        nodeId,
        NodeGraphParamValue{nodegraphparams::merge::MergeDistance, NodeGraphParamType::Float, static_cast<double>(params.mergeDistance)});
}
