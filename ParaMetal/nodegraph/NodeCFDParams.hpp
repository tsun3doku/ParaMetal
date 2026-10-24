#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/CFDData.hpp"
#include <array>
#include <glm/glm.hpp>

class NodeGraphEditor;
struct NodeGraphNode;

struct CFDNodeParams {
    bool enabled = true;
    double targetCourant = 0.5;
    double timeStep = 1.0 / 60.0;
    glm::vec3 initialVelocity{0.0f};
    double pressureIterations = 200.0;
    double pressureTolerance = 1e-4;

    std::array<CFDData::BoundaryCondition, 6> boundaries{};
};

CFDNodeParams readCFDNodeParams(const NodeGraphNode& node);
bool writeCFDNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const CFDNodeParams& params);
