#include "NodeGraphDisplay.hpp"

#include "NodeGraphDataTypes.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphProductTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeECS.hpp"

#include <unordered_set>
#include <vector>

std::unordered_set<uint64_t> NodeGraphDisplay::computeDisplayKeys(
    const NodeGraphState& graphState,
    const NodeGraphEvaluationState& evaluationState,
    const ECSRegistry& registry,
    const NodePayloadRegistry* payloadRegistry) const {

    std::unordered_set<uint64_t> selectedKeys;

    for (const auto& [id, node] : graphState.nodes) {
        if (!node.displayEnabled) {
            continue;
        }

        for (const NodeGraphSocket& output : node.outputs) {
            const uint64_t socketKey = NodeSocketKey(node.id, output.id);
            const auto valueIt = evaluationState.outputBySocket.find(socketKey);
            const NodeDataBlock* block = (valueIt != evaluationState.outputBySocket.end() && valueIt->second.status == EvaluatedSocketStatus::Value)
                ? &valueIt->second.data
                : nullptr;
            addDisplayKeys(socketKey, block, registry, payloadRegistry, selectedKeys);
        }
    }

    return selectedKeys;
}

void NodeGraphDisplay::addDisplayKeys(
    uint64_t socketKey,
    const NodeDataBlock* block,
    const ECSRegistry& registry,
    const NodePayloadRegistry* payloadRegistry,
    std::unordered_set<uint64_t>& selectedKeys) const {

    if (socketKey == 0) {
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);

    // Mesh-like payloads: Geometry, Remesh, HeatModel
    if (registry.all_of<ModelPackage>(entity)) {
        selectedKeys.insert(socketKey);
        return;
    }

    if (registry.all_of<RemeshPackage>(entity)) {
        selectedKeys.insert(socketKey);
        const auto& package = registry.get<RemeshPackage>(entity);
        if (package.sourceMeshHandle.key != 0) {
            selectedKeys.insert(package.sourceMeshHandle.key);
        }
        return;
    }

    if (block && block->dataType == payloadtypes::HeatModel && payloadRegistry) {
        NodeDataHandle sourceModelHandle{};
        payloadRegistry->resolveGeometry(block->payloadHandle, &sourceModelHandle);
        const NodeDataHandle currentMeshHandle = payloadRegistry->resolveMeshHandle(block->dataType, block->payloadHandle);
        if (sourceModelHandle.key != 0) {
            selectedKeys.insert(sourceModelHandle.key);
        }
        if (currentMeshHandle.key != 0 &&
            currentMeshHandle.key != sourceModelHandle.key &&
            registry.valid(static_cast<ECSEntity>(currentMeshHandle.key)) &&
            registry.all_of<RemeshPackage>(static_cast<ECSEntity>(currentMeshHandle.key))) {
            selectedKeys.insert(currentMeshHandle.key);
        }
        return;
    }

    // Voronoi
    if (registry.all_of<VoronoiPackage>(entity)) {
        selectedKeys.insert(socketKey);
        const auto& package = registry.get<VoronoiPackage>(entity);
        if (package.modelMeshHandle.key != 0) {
            selectedKeys.insert(package.modelMeshHandle.key);
        }
        if (package.modelRemeshHandle.key != 0) {
            selectedKeys.insert(package.modelRemeshHandle.key);
        }
        return;
    }

    // Contact
    if (registry.all_of<ContactPackage>(entity)) {
        selectedKeys.insert(socketKey);
        const auto& package = registry.get<ContactPackage>(entity);

        auto addEndpointKeys = [&](const NodeDataHandle& meshHandle) {
            if (meshHandle.key == 0 || !payloadRegistry) {
                return;
            }
            NodeDataHandle sourceModelHandle{};
            payloadRegistry->resolveGeometry(meshHandle, &sourceModelHandle);
            const NodeDataHandle currentMeshHandle = payloadRegistry->resolveMeshHandle(
                payloadtypes::Remesh, meshHandle);
            if (sourceModelHandle.key != 0) {
                selectedKeys.insert(sourceModelHandle.key);
            }
            if (currentMeshHandle.key != 0 &&
                currentMeshHandle.key != sourceModelHandle.key &&
                registry.valid(static_cast<ECSEntity>(currentMeshHandle.key)) &&
                registry.all_of<RemeshPackage>(static_cast<ECSEntity>(currentMeshHandle.key))) {
                selectedKeys.insert(currentMeshHandle.key);
            }
        };

        addEndpointKeys(package.modelAMeshHandle);
        addEndpointKeys(package.modelBMeshHandle);
        return;
    }

    // Heat
    if (registry.all_of<HeatPackage>(entity)) {
        selectedKeys.insert(socketKey);
        const auto& package = registry.get<HeatPackage>(entity);
        for (const NodeDataHandle& handle : package.resolvedModelHandles) {
            if (handle.key != 0) {
                selectedKeys.insert(handle.key);
            }
        }
        for (const NodeDataHandle& handle : package.resolvedRemeshHandles) {
            if (handle.key != 0) {
                selectedKeys.insert(handle.key);
            }
        }
        for (const NodeDataHandle& handle : package.authored.voronoiHandles) {
            if (handle.key != 0) {
                selectedKeys.insert(handle.key);
            }
        }
        for (const NodeDataHandle& handle : package.authored.contactHandles) {
            if (handle.key != 0) {
                selectedKeys.insert(handle.key);
            }
        }
        return;
    }
}
