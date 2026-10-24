#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace voronoi {

struct SurfaceVertex {
    glm::vec4 position{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 normal{0.0f, 0.0f, 1.0f, 0.0f};
};

static_assert(sizeof(SurfaceVertex) == 32, "SurfaceVertex must match GPU stride");

struct CandidatePushConstants {
    uint32_t faceCount;
    uint32_t seedCount;
    uint32_t _pad0;
    uint32_t _pad1;
};

static_assert(sizeof(CandidatePushConstants) == 16, "CandidatePushConstants must match GPU stride");

inline constexpr uint32_t InvalidModelId = 0xffffffffu;

struct Fragment {
    uint32_t cellId = 0;
    uint32_t ownerModelId = InvalidModelId;
    double volume = 0.0;
    glm::dvec3 centroid{0.0};
};

struct ContactMeshVertex {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f};
};

}