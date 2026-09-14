#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "heat/HeatSystemPresets.hpp"

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
    NodeDataHandle domainVoronoiHandle;
    std::vector<NodeDataHandle> heatModelHandles;
    NodeDataHandle heatFluidHandle{};
    uint32_t activeVoronoiCount = 0;
    float simulationDuration = 5.0f;
    bool hasFluid = false;
    bool active = false;

    uint32_t domainBoundaryTypes[6] = {};
    float domainBoundaryTemperaturesC[6] = {
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC
    };
    float domainBoundaryHeatFluxes[6] = {};
    float domainBoundaryHeatTransferCoefficients[6] = {};
};
