#pragma once

#include "domain/GeometryData.hpp"

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - They must not be used directly by any backends ]

struct RemeshData {
    NodeDataHandle sourceMeshHandle{};
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    bool active = false;

    // Completed surface mesh. Empty until the Remesh product is committed.
    std::vector<glm::vec3> surfacePositions;
    std::vector<uint32_t> surfaceTriangleIndices;
    // One raw intrinsic ID per compact surface ordinal.
    std::vector<uint32_t> surfaceFaceIds;
    std::vector<uint32_t> surfaceVertexIds;
    std::vector<uint32_t> surfaceEdgeIds;
    // Publication identity of the Remesh product that produced this surface.
    uint64_t surfaceGeometryHash = 0;
};
