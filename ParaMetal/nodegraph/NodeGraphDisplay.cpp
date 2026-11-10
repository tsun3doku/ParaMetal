#include "NodeGraphDisplay.hpp"

#include "NodeGraphDataTypes.hpp"
#include "NodeGraphEvaluation.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphProductTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"

#include <unordered_set>

std::unordered_set<uint64_t> NodeGraphDisplay::computeDisplayKeys(
    const NodeGraphState& graphState,
    const NodeGraphEvaluation& evaluation,
    const RuntimePackageManager& packages,
    const NodePayloadRegistry* payloadRegistry) const {

    std::unordered_set<uint64_t> selectedKeys;

    for (const auto& [id, node] : graphState.nodes) {
        if (!node.state.isPrimaryDisplay()) {
            continue;
        }

        for (const NodeGraphSocket& output : node.outputs) {
            const uint64_t socketKey = NodeSocketKey(node.id, output.id);
            const EvaluatedSocketValue* value = evaluation.outputFor(socketKey);
            const NodeDataBlock* block = (value && value->status == EvaluatedSocketStatus::Value) ? &value->data : nullptr;

            addDisplayKeys(socketKey, block, packages, payloadRegistry, selectedKeys);
        }
    }

    return selectedKeys;
}

void NodeGraphDisplay::addDisplayKeys(
    uint64_t socketKey,
    const NodeDataBlock* block,
    const RuntimePackageManager& packages,
    const NodePayloadRegistry* payloadRegistry,
    std::unordered_set<uint64_t>& selectedKeys) const {

    if (socketKey == 0) {
        return;
    }

    // Packages declare the sockets their own view makes visible (upstream models,
    // remeshes, domain products). NodeGraphDisplay only selects and collects them.
    if (const std::vector<uint64_t>* dependencyKeys =
            packages.displayDependencyKeysFor(socketKey)) {
        selectedKeys.insert(socketKey);
        selectedKeys.insert(dependencyKeys->begin(), dependencyKeys->end());
        return;
    }

    // Fallback for non-package authoring nodes (e.g. previewing a HeatModel node directly)
    if (block && block->dataType == payloadtypes::HeatModel && payloadRegistry) {
        NodeDataHandle currentMeshHandle{};
        if (payloadRegistry->resolveRemesh(block->payloadHandle, &currentMeshHandle) &&
            currentMeshHandle.key != 0) {
            if (const RemeshPackage* remesh = packages.findAny<RemeshPackage>(currentMeshHandle.key)) {
                selectedKeys.insert(currentMeshHandle.key);
                if (remesh->sourceModelProduct.isValid()) {
                    selectedKeys.insert(remesh->sourceModelProduct.outputSocketKey);
                }
            } else {
                selectedKeys.insert(currentMeshHandle.key);
            }
        }
        return;
    }
}
