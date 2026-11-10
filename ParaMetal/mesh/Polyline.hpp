#pragma once

#include <glm/vec3.hpp>

#include <vector>

namespace mesh {

struct Polyline {
    std::vector<glm::vec3> points;
    bool closed = false;
};

} // namespace mesh
