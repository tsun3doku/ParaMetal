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
    std::vector<NodeDataHandle> voronoiHandles;
    std::vector<NodeDataHandle> contactHandles;
    std::vector<NodeDataHandle> heatModelHandles;
    float contactThermalConductance = 16000.0f;
    float simulationDuration = 5.0f;
    bool active = false;
    bool paused = false;
    uint32_t resetCounter = 0;
    uint32_t rewindFrame = std::numeric_limits<uint32_t>::max();

};
