#pragma once

#include "hash/HashValues.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects, 
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

enum class DomainType : uint8_t { Mesh, Points };

struct VoronoiData {
    HashValues hashes{};
    float cellSize = 0.005f;
    int voxelResolution = 128;
    DomainType domainType = DomainType::Mesh;

    // Mesh path
    NodeDataHandle modelMeshHandle;

    // Point path
    NodeDataHandle pointsPayloadHandle;

    bool active = false;

};
