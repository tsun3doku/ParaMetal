#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects, 
//                                                          backend/controller objects, or GPU resources
//                                                        - This header must not be included in any backend ]

struct VoronoiData {
    uint64_t payloadHash = 0;
    float cellSize = 0.005f;
    int voxelResolution = 128;
    std::vector<NodeDataHandle> modelMeshHandles;
    std::vector<uint64_t> modelPayloadHashes;
    std::vector<NodeDataHandle> modelPayloadHandles; // Original payload handles (GeometryData or HeatModelData)
    bool active = false;

    void sealPayload();
};
