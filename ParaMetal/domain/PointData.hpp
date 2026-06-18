#pragma once

#include "hash/HashValues.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <array>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects, 
//                                                          backend/controller objects or GPU resources 
//                                                        - They must not be used directly by any backends ]

struct PointData {
    std::vector<glm::vec4> positions;  // xyz local space, w = 1
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    HashValues hashes{};
    bool active = false;

};
