#pragma once

#include "util/Units.hpp"

#include <string>
#include <vector>

class NodePayloadRegistry;
class RuntimePackageManager;
class RuntimeProductManager;
struct NodeGraphCompiled;
struct NodeGraphEvaluation;
struct NodeGraphNode;
struct NodeGraphState;

class RuntimePackageCompiler {
public:
    enum class FrozenPackages {
        Preserve,
        Rebuild,
    };

    bool validate(
        const NodeGraphState& graph,
        const NodeGraphCompiled& graphPlan,
        const NodeGraphEvaluation& evaluation,
        const NodePayloadRegistry& payloads,
        std::vector<std::string>& errors) const;

    bool compileNode(
        const NodeGraphState& graph,
        const NodeGraphNode& node,
        const NodeGraphEvaluation& evaluation,
        const NodePayloadRegistry& payloads,
        units::LengthUnit worldUnit,
        FrozenPackages frozenPackages,
        RuntimePackageManager& packages,
        const RuntimeProductManager* products,
        std::vector<std::string>& errors) const;
};
