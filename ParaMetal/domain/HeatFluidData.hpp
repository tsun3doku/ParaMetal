#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "heat/HeatSystemPresets.hpp"
#include <glm/glm.hpp>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct HeatFluidData {
    float density = 1.225f;
    float specificHeat = 1005.0f;
    float conductivity = 0.026f;
    float initialTemperatureC = HeatSimDefaults::ambientTemperatureC;
    float inletTemperatureC = HeatSimDefaults::ambientTemperatureC;
    glm::vec3 velocity{0.0f};
};
