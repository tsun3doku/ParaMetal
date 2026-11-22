#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nodegraph/NodeGraphDataTypes.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "util/Units.hpp"

class NodePayloadRegistry;
class RuntimePackageManager;

namespace runtimepackage {

struct PackageCompileContext {
    const NodeGraphState& graph;
    const NodeGraphNode& node;
    const NodeDataBlock& output;
    uint64_t outputSocketKey = 0;
    const NodePayloadRegistry& payloads;
    units::LengthUnit worldUnit = units::defaultLengthUnit();
    RuntimePackageManager& packages;
    std::vector<std::string>& errors;
};

}
