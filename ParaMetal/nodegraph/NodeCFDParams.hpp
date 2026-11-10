#pragma once

#include "NodeGraphTypes.hpp"
#include <glm/glm.hpp>

class NodeGraphEditor;
struct NodeGraphNode;

struct CFDPreviewSettings {
    bool showStreamlines = false;
    bool showDomainBox = false;
    bool showVelocityPalette = false;
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
bool writeCFDNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const CFDNodeParams& params);
