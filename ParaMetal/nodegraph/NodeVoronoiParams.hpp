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

namespace nodegraphparams::voronoi {
constexpr uint32_t VoxelResolution = 2;
constexpr uint32_t ShowVoronoi = 3;
constexpr uint32_t ShowPoints = 4;
constexpr uint32_t VolumePadding = 5;
constexpr uint32_t ContactRange = 6;
constexpr uint32_t LloydIterations = 7;
constexpr uint32_t LloydVolumeThreshold = 9;
constexpr uint32_t LloydDamping = 10;
}
