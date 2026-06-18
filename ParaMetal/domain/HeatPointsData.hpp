#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <cstdint>
#include <vector>

struct HeatPointsData {
    NodeDataHandle pointsHandle;
    std::vector<uint32_t> boundaryConditions;
    std::vector<float> fixedTemperatures;
    float initialTemperature = 293.15f;
    bool active = false;

};
