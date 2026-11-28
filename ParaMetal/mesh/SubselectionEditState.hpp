#pragma once

#include "mesh/MeshSelection.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

struct SubselectionEditState {
    uint64_t sourceSocketKey = 0;
    uint64_t displaySocketKey = 0;
    uint64_t surfaceIdentity = 0;
    uint64_t pickGeometryIdentity = 0;
    SurfaceBinding binding = SurfaceBinding::Source;

    uint32_t runtimeModelId = 0;
    uint32_t faceCount = 0;

    glm::mat4 modelMatrix{1.0f};
    glm::mat4 viewProjection{1.0f};
    glm::vec2 viewportSize{0.0f};
    std::vector<uint32_t> overlayIndices;
    std::vector<glm::vec3> vertexPositions;
    std::vector<uint32_t> triangleIndices;
    std::vector<uint32_t> surfaceFaceIds;
    std::vector<uint32_t> surfaceVertexIds;
    std::vector<uint32_t> surfaceEdgeIds;
};
