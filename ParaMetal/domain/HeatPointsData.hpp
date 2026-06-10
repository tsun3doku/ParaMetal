#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <cstdint>
#include <vector>

struct HeatPointsData {
    NodeDataHandle pointsHandle;
    std::vector<uint32_t> boundaryConditions;
    std::vector<float> fixedTemperatures;
    float initialTemperature = 293.15f;
    uint64_t payloadHash = 0;
    bool active = false;

    void sealPayload();
};
