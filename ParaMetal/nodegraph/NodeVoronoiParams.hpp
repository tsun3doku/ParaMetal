#pragma once

#include "NodeGraphTypes.hpp"
#include "voronoi/VoronoiConfig.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct VoronoiPreviewSettings {
    bool showVoronoi = false;
    bool showPoints = false;
};

struct VoronoiNodeParams {
    double cellSize = 0.005;
    int voxelResolution = 128;
    double volumePadding = 0.01;
    int lloydIterations = int(voronoi::DefaultLloydIterations);
    double lloydVolumeThreshold = voronoi::DefaultLloydVolumeThresholdFraction;
    double lloydDamping = voronoi::DefaultLloydDamping;
    double contactRange = voronoi::DefaultContactRange;
    VoronoiPreviewSettings preview{};
};

VoronoiNodeParams readVoronoiNodeParams(const NodeGraphNode& node);
bool writeVoronoiNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const VoronoiNodeParams& params);
