#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

struct PointData {
    std::vector<glm::vec4> positions;  // xyz local space, w = 1
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    uint64_t payloadHash = 0;
    bool active = false;

    void sealPayload();
};
