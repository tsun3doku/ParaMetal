#pragma once

#include "nodegraph/NodeInteraction.hpp"

#include <string>
#include <variant>

struct NodeInspectionChanged {
    uint64_t graphInstanceId = 0;
    NodeGraphNodeId nodeId{};
    NodeInspection inspection;
};

struct NodeActionCompleted {
    NodeGraphNodeId nodeId{};
    bool success = false;
    std::string error;
};

struct RuntimeNodeSelectionRequested {
    uint64_t graphInstanceId = 0;
    NodeGraphNodeId nodeId{};
    uint64_t nodeInstanceId = 0;
    bool additive = false;
};

using RuntimeEvent = std::variant<
    NodeInspectionChanged,
    NodeActionCompleted,
    RuntimeNodeSelectionRequested
>;
