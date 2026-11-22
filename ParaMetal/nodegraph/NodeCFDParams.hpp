#pragma once

#include "NodeGraphTypes.hpp"
#include <glm/glm.hpp>

struct NodeGraphNode;

struct CFDPreviewSettings {
    bool showStreamlines = false;
    bool showDomainBox = false;
    bool showVelocityPalette = false;
    bool showSlice = false;
    int sliceAxis = 0;
    double slicePosition = 0.5;
    bool showSliceWireframe = false;
};

struct CFDNodeParams {
    bool enabled = true;
    double targetCourant = 0.5;
    double timeStep = 1.0 / 60.0;
    glm::vec3 initialVelocity{0.0f};
    double pressureIterations = 200.0;
    double pressureTolerance = 1e-4;
    double simulationDuration = 10.0;
    CFDPreviewSettings preview{};
};

CFDNodeParams readCFDNodeParams(const NodeGraphNode& node);

namespace nodegraphparams::cfd {
constexpr uint32_t Enabled = 1;
constexpr uint32_t TargetCourant = 2;
constexpr uint32_t TimeStep = 3;
constexpr uint32_t VelocityX = 4;
constexpr uint32_t VelocityY = 5;
constexpr uint32_t VelocityZ = 6;
constexpr uint32_t PressureIterations = 7;
constexpr uint32_t PressureTolerance = 8;
constexpr uint32_t Duration = 9;
constexpr uint32_t ShowDomainBox = 24;
constexpr uint32_t ShowStreamlines = 28;
constexpr uint32_t ShowVelocityPalette = 29;
constexpr uint32_t ShowSlice = 30;
constexpr uint32_t SliceAxis = 31;
constexpr uint32_t SlicePosition = 32;
constexpr uint32_t ShowSliceWireframe = 35;
}
