#pragma once

#include "NodeGraphTypes.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct MergeNodeParams {
    float mergeDistance = 0.01f;
};

MergeNodeParams readMergeNodeParams(const NodeGraphNode& node);
bool writeMergeNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const MergeNodeParams& params);
