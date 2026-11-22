#pragma once

#include "NodeGraphTypes.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct RemeshPreviewSettings {
    bool showRemeshOverlay = false;
    bool showFaceNormals = false;
    bool showVertexNormals = false;
};

struct RemeshNodeParams {
    int iterations = 1;
    double minAngleDegrees = 20.0;
    double maxEdgeLength = 0.1;
    double stepSize = 0.25;
    RemeshPreviewSettings preview{};
    double normalLength = 0.05;
};

RemeshNodeParams readRemeshNodeParams(const NodeGraphNode& node);
bool writeRemeshNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const RemeshNodeParams& params);

namespace nodegraphparams::remesh {
constexpr uint32_t Iterations = 1;
constexpr uint32_t MinAngleDegrees = 2;
constexpr uint32_t MaxEdgeLength = 3;
constexpr uint32_t StepSize = 4;
constexpr uint32_t RunRequested = 5;
constexpr uint32_t ShowRemeshOverlay = 6;
constexpr uint32_t ShowFaceNormals = 7;
constexpr uint32_t ShowVertexNormals = 8;
constexpr uint32_t NormalLength = 9;
}
