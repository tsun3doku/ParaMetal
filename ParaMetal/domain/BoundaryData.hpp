#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct BoundaryData {
    enum class FlowType : uint8_t {
        NoSlipWall,
        VelocityInlet,
        PressureOutlet
    };

    enum class ThermalType : uint8_t {
        Adiabatic,
        Dirichlet,
        Neumann,
        Robin
    };

    struct Face {
        FlowType flowType = FlowType::NoSlipWall;
        glm::vec3 velocity{0.0f};
        ThermalType thermalType = ThermalType::Adiabatic;
        float temperatureC = 5.0f;
        float heatFlux = 0.0f;
        float htc = 0.0f;
        float inflowTemperatureC = 20.0f;
    };

    std::array<Face, 6> faces{};
};
