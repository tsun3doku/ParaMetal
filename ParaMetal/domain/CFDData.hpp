#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <glm/glm.hpp>

struct CFDData {
    NodeDataHandle domainVoronoiHandle{};
    NodeDataHandle fluidHandle{};
    NodeDataHandle boundaryHandle{};

    bool hasFluid = false;
    bool active = false;

    float fixedTimeStep = 1.0f / 60.0f;
    float targetCourant = 0.5f;
    int pressureIterations = 0;
    float pressureTolerance = 0.0f;
    float simulationDuration = 10.0f;

    glm::vec3 initialVelocity{0.0f};
};
