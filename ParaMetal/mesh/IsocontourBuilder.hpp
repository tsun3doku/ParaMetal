#pragma once

#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

struct IsocontourPolyline {
    std::vector<glm::vec3> points;
    float isoValue = 0.0f;
    bool closed = false;
};

class IsocontourBuilder {
public:
    static std::vector<IsocontourPolyline> build(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& triangleIndices,
        const std::vector<float>& scalarValues,
        const std::vector<float>& isoValues);
};
