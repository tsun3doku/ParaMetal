#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nodegraph/NodeGraphEvaluation.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "util/Units.hpp"

class NodePayloadRegistry;
class RuntimePackageManager;
class RuntimeProductManager;

namespace runtimepackage {

struct PackageCompileContext {
    const NodeGraphState& graph;
    const NodeGraphNode& node;
    const NodeDataBlock& output;
    uint64_t outputSocketKey = 0;
    const NodePayloadRegistry& payloads;
    const NodeGraphEvaluation& evaluation;
    units::LengthUnit worldUnit = units::defaultLengthUnit();
    RuntimePackageManager& packages;
    std::vector<std::string>& errors;
    const RuntimeProductManager* products = nullptr;

    const NodeDataBlock* input(NodeGraphValueType type) const {
        const std::size_t index = inputIndexOf(node, type);
        if (index >= node.inputs.size()) return nullptr;
        const NodeGraphEdge* edge = graph.edges.incomingEdge(node.id, node.inputs[index].id);
        if (!edge) return nullptr;
        const EvaluatedSocketValue* value = evaluation.outputFor(
            NodeSocketKey(edge->fromNode, edge->fromSocket).value);
        return value && value->status == EvaluatedSocketStatus::Value
            ? &value->data : nullptr;
    }
};

}
