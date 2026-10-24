#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct HeatData {
    enum class BoundaryType : uint32_t {
        Adiabatic = 0,
        Dirichlet = 1, // Fixed temperature
        Neumann   = 2, // Fixed heat flux
        Robin     = 3  // Convective (HTC)
    };

    struct BoundaryCondition {
        BoundaryType type = BoundaryType::Adiabatic;
        float temperatureC = 20.0f;
        float heatFlux = 0.0f;
        float htc = 0.0f;
    };

    NodeDataHandle domainVoronoiHandle;
    std::vector<NodeDataHandle> heatModelHandles;
    NodeDataHandle heatFluidHandle{};
    NodeDataHandle cfdHandle{};
    uint32_t activeVoronoiCount = 0;
    float simulationDuration = 5.0f;
    uint32_t solverMaxIterations = 0;         // 0 = keep heat solver default
    float solverConvergenceTolerance = 0.0f;  // 0 = keep heat solver default
    bool hasFluid = false;
    bool active = false;

    std::array<BoundaryCondition, 6> boundaries{
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f},
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f},
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f},
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f},
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f},
        BoundaryCondition{BoundaryType::Adiabatic, 20.0f, 0.0f, 0.0f}
    };
};
