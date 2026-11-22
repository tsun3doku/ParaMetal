#pragma once

#include "NodeGraphTypes.hpp"

class NodeGraph;
class NodeGraphEditor;
struct NodeGraphNode;

struct TransformNodeParams {
    double translateX = 0.0;
    double translateY = 0.0;
    double translateZ = 0.0;
    double rotateXDegrees = 0.0;
    double rotateYDegrees = 0.0;
    double rotateZDegrees = 0.0;
    double scaleX = 1.0;
    double scaleY = 1.0;
    double scaleZ = 1.0;
};

TransformNodeParams readTransformNodeParams(const NodeGraphNode& node);
bool writeTransformNodeParams(NodeGraph& bridge, NodeGraphNodeId nodeId, const TransformNodeParams& params);
bool writeTransformNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const TransformNodeParams& params);

namespace nodegraphparams::transform {
constexpr uint32_t TranslateX = 1;
constexpr uint32_t TranslateY = 2;
constexpr uint32_t TranslateZ = 3;
constexpr uint32_t RotateXDegrees = 4;
constexpr uint32_t RotateYDegrees = 5;
constexpr uint32_t RotateZDegrees = 6;
constexpr uint32_t ScaleX = 7;
constexpr uint32_t ScaleY = 8;
constexpr uint32_t ScaleZ = 9;
}
