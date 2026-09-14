#pragma once

#include <cstdint>
#include <cstddef>
#include <limits>

#include <glm/glm.hpp>

namespace heat {

inline constexpr uint32_t NoRewindFrame = std::numeric_limits<uint32_t>::max();

struct SimPlaybackUniform {
    uint32_t resetCounter;
    uint32_t recordedTimelineFrames;
    uint32_t timelineFrameCount;
    float deltaTime;
};

struct SurfacePoint {
    glm::vec3 position;
    float temperatureC;
    glm::vec3 normal;
    float vertexArea;
    glm::vec4 color;
};

struct HeatModelPushConstant {
    uint32_t elementCount;
    float metersPerUnit;
};
static_assert(sizeof(HeatModelPushConstant) == 8);


struct GMLSSurfaceStencil {
    uint32_t valueWeightOffset;
    uint32_t valueWeightCount;
    uint32_t gradientWeightOffset;
    uint32_t gradientWeightCount;
};
static_assert(sizeof(GMLSSurfaceStencil) == 16, "GMLSSurfaceStencil must match GPU stride");

struct GMLSSurfaceWeight {
    uint32_t cellIndex;
    float weight;
};
static_assert(sizeof(GMLSSurfaceWeight) == 8, "GMLSSurfaceWeight must match GPU stride");

struct GMLSSurfaceGradientWeight {
    uint32_t cellIndex;
    float dTdxWeight;
    float dTdyWeight;
    float dTdzWeight;
};
static_assert(sizeof(GMLSSurfaceGradientWeight) == 16, "GMLSSurfaceGradientWeight must match GPU stride");

struct AdvectionFace {
    uint32_t nodeA = 0;
    uint32_t nodeB = 0;
    float invVolA = 0.0f;
    float invVolB = 0.0f;
};
static_assert(sizeof(AdvectionFace) == 16, "AdvectionFace must match GPU stride");

struct AdvectionBoundaryFace {
    uint32_t node = 0;
    float invVol = 0.0f;
};
static_assert(sizeof(AdvectionBoundaryFace) == 8, "AdvectionBoundaryFace must match GPU stride");

}

