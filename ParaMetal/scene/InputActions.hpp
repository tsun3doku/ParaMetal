#pragma once

#include "nodegraph/NodeGraphTypes.hpp"

#include <variant>
#include <vector>

struct ToggleWireframeAction {};
struct ToggleTimingOverlayAction {};
struct ToggleGridAction {};

struct SetNodeParametersAction {
    NodeGraphNodeId nodeId;
    std::vector<NodeGraphParamValue> parameters;
};

using InputAction = std::variant<
    ToggleWireframeAction,
    ToggleTimingOverlayAction,
    ToggleGridAction,
    SetNodeParametersAction>;
