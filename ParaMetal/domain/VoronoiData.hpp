#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "voronoi/VoronoiConfig.hpp"

#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct VoronoiData {
    float cellSize = 0.005f;
    int voxelResolution = 128;
    float volumePadding = 0.01f;

    NodeDataHandle pointsPayloadHandle;

    std::vector<NodeDataHandle> modelMeshHandles;

    bool active = false;
    float contactRange = voronoi::DefaultContactRange;
};
