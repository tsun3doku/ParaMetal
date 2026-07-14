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

struct BoundaryCondition {
    enum class Type : uint32_t {
        Adiabatic = 0,
        DirichletTemperature = 1,
        NeumannHeatFlux = 2,
        RobinConvection = 3,
    };

    Type type = Type::Adiabatic;
    float temperatureC = HeatSimDefaults::ambientTemperatureC;
    float heatFlux = 0.0f;
    float heatTransferCoefficient = 0.0f;
};

struct VolumetricHeatSource {
    float powerDensity = 0.0f;
};

struct HeatModelData {
    NodeDataHandle meshHandle{};

    // Material
    float density = HeatSimDefaults::density;
    float specificHeat = HeatSimDefaults::specificHeat;
    float conductivity = HeatSimDefaults::conductivity;

    // State
    float initialTemperatureC = HeatSimDefaults::ambientTemperatureC;

    // Boundary condition
    BoundaryCondition boundaryCondition;

    // Whole-model volumetric load
    VolumetricHeatSource volumetricHeatSource;

};
