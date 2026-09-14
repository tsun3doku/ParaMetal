#pragma once

#include "NodeGraphTypes.hpp"
#include <cstdint>

class NodeGraphEditor;
struct NodeGraphNode;

enum class PointsDistribution : uint32_t {
    UniformGrid = 0,
    BlueNoise = 1
};

struct PointsNodeParams {
    uint32_t pointCount = 1000;
    float dimX = 1.0f;
    float dimY = 1.0f;
    float dimZ = 1.0f;

    PointsDistribution distribution = PointsDistribution::BlueNoise;
    uint32_t seed = 1;
};

PointsNodeParams readPointsNodeParams(const NodeGraphNode& node);
bool writePointsNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const PointsNodeParams& params);
