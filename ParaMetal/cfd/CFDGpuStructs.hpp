#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace cfd {

struct CFDFaceGpu {
    uint32_t cellA = 0;
    uint32_t cellB = 0;
    glm::vec3 fluxVectorA{0.0f};
    glm::vec3 fluxVectorB{0.0f};
    float projectionWeight = 0.0f; // sum of constituent projection weights
};
static_assert(sizeof(CFDFaceGpu) == 36, "CFDFaceGpu must match GPU stride");

struct CFDBoundaryFaceGpu {
    uint32_t cell = 0;
    uint32_t boundaryCondition = UINT32_MAX; // 0..5 domain boundary, 6 embedded solid wall
    glm::vec3 areaVector{0.0f};
    float projectionWeight = 0.0f; // sum of constituent (area / distance)
};
static_assert(sizeof(CFDBoundaryFaceGpu) == 24, "CFDBoundaryFaceGpu must match GPU stride");

struct BoundaryConditionGpu {
    uint32_t type = 0;
    glm::vec3 velocity{0.0f};
};
static_assert(sizeof(BoundaryConditionGpu) == 16, "BoundaryConditionGpu must match GPU stride");

struct StreamlinePushConstants {
    glm::vec4 gridMinCellSize{0.0f};
    glm::ivec4 gridDimMaxSegments{0};
    glm::vec4 stepLengthPadding{0.0f};
    glm::uvec4 counts{0u};
    glm::vec4 domainMin{0.0f};
    glm::vec4 domainMax{0.0f};
};
static_assert(sizeof(StreamlinePushConstants) == 96, "StreamlinePushConstants must match GPU stride");

struct LineVertex {
    glm::vec4 positionSpeed{0.0f};
};
static_assert(sizeof(LineVertex) == 16, "LineVertex must match GPU stride");

}
