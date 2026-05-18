#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "heat/HeatSystemPresets.hpp"

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

enum class HeatBoundaryCondition {
    None,              // Normal thermal mass 
    FixedTemperature,  // Dirichlet BC 
    FixedPower         // Neumann BC 
};

struct HeatModelData {
    uint64_t payloadHash = 0;
    NodeDataHandle meshHandle{};
    uint64_t meshPayloadHash = 0;

    // Material
    float density = HeatSimDefaults::density;
    float specificHeat = HeatSimDefaults::specificHeat;
    float conductivity = HeatSimDefaults::conductivity;

    // State
    float initialTemperature = HeatSimDefaults::ambientTemperature;

    // Boundary condition
    HeatBoundaryCondition boundaryCondition = HeatBoundaryCondition::None;
    float fixedTemperatureValue = HeatSimDefaults::ambientTemperature;

    void sealPayload();
};
