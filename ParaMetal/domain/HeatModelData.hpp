#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <cstdint>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct VolumetricHeatSource {
    float powerDensity = 0.0f;
};

struct HeatModelData {
    NodeDataHandle meshHandle{};

    float density = HeatSimDefaults::density;
    float specificHeat = HeatSimDefaults::specificHeat;
    float conductivity = HeatSimDefaults::conductivity;
    float initialTemperatureC = HeatSimDefaults::ambientTemperatureC;

    VolumetricHeatSource volumetricHeatSource;
    bool prescribedTemperatureEnabled = false;
    float prescribedTemperatureC = HeatSimDefaults::ambientTemperatureC;
};
