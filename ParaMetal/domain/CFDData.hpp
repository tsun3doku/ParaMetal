#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

struct CFDData {
    enum class BoundaryType : uint8_t {
        NoSlipWall,
        VelocityInlet,
        PressureOutlet
    };

    struct BoundaryCondition {
        BoundaryType type = BoundaryType::NoSlipWall;
        glm::vec3 velocity{0.0f};
    };

    NodeDataHandle domainVoronoiHandle{};
    NodeDataHandle fluidHandle{};

    bool hasFluid = false;
    bool active = false;

    float fixedTimeStep = 1.0f / 60.0f;
    float targetCourant = 0.5f;
    int pressureIterations = 0;
    float pressureTolerance = 0.0f;

    glm::vec3 initialVelocity{0.0f};

    std::array<BoundaryCondition, 6> boundaries{};
};
