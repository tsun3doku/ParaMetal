#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include <cstdint>
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
    NodeDataHandle cfdHandle{};
    NodeDataHandle boundaryHandle{};
    uint32_t activeVoronoiCount = 0;
    float simulationDuration = 5.0f;
    uint32_t solverMaxIterations = 0;         // 0 = keep heat solver default
    float solverConvergenceTolerance = 0.0f;  // 0 = keep heat solver default
    float isocontourSpacingC = 10.0f;
    bool hasFluid = false;
    bool active = false;
};
