#pragma once

#include "util/Units.hpp"

#include <string>
#include <vector>

class NodePayloadRegistry;
class RuntimePackageManager;
class RuntimePackageController;
class NodeGraphRuntime;
class RuntimeProductManager;
struct NodeGraphEvaluation;
struct NodeGraphNode;
struct NodeGraphState;

class RuntimePackageCompiler {
public:
    enum class FrozenPackages {
        Preserve,
        Rebuild,
    };

    // Evaluate/compile/apply/publish is one ordered node transaction.
    bool executeNode(
        const NodeGraphState& graph,
        const NodeGraphNode& node,
        NodeGraphRuntime& graphRuntime,
        NodePayloadRegistry& payloads,
        units::LengthUnit worldUnit,
        FrozenPackages frozenPackages,
        RuntimePackageController& backend,
        std::vector<std::string>& errors,
        bool& backpressured) const;
};
