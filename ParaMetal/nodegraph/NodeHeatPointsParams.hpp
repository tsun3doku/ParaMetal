#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatModelData.hpp"

#include <vector>

#include <glm/vec3.hpp>

class NodeGraphEditor;
struct NodeGraphNode;

struct HeatPointNodeRow {
    uint32_t boundaryCondition = 0;
    float fixedTemperature = 293.15f;
};

struct HeatPointsNodeParams {
    std::vector<HeatPointNodeRow> rows;
    double initialTemperature = 293.15;
    double fixedTemperature = 293.15;
    HeatBoundaryCondition boundaryCondition = HeatBoundaryCondition::None;
};

HeatPointsNodeParams readHeatPointsNodeParams(const NodeGraphNode& node);
bool writeHeatPointsNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatPointsNodeParams& params);
