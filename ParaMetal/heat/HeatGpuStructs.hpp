#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace heat {

struct SimPlaybackUniform {
    uint32_t paused;
    uint32_t resetCounter;
    uint32_t rewindFrame;
    uint32_t recordedFrames;
    uint32_t maxFrames;
    float deltaTime;
    float totalTime;
    uint32_t _pad;
};

struct SurfacePoint {
    glm::vec3 position;
    float temperature;
    glm::vec3 normal;
    float area;
    glm::vec4 color;
};

struct HeatModelPushConstant {
    uint32_t elementCount;
};

struct MaterialNode {
    float conductivityPerMass;
    float thermalMass;
    float density;
    float specificHeat;
    float conductivity;
    float fixedTemperatureValue;
    uint32_t boundaryCondition;
    uint32_t _pad0;
};

struct BufferPushConstant {
    alignas(16) glm::mat4 modelMatrix;
    alignas(16) glm::vec4 sourceParams;
};

struct SourceRenderPushConstant {
    alignas(16) glm::mat4 modelMatrix;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 sourceParams;
};

}
