#pragma once

#include "Polyline.hpp"

#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

class IsocontourBuilder {
public:
    static std::vector<mesh::Polyline> build(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& triangleIndices,
        const std::vector<float>& scalarValues,
        const std::vector<float>& isoValues);
};
