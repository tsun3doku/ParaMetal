#include "NodeRemesh.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphBridge.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeRemeshParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

const char* NodeRemesh::typeId() const {
    return nodegraphtypes::Remesh;
}

void NodeRemesh::execute(NodeGraphKernelContext& context) const {
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    RemeshData remeshData{};

    const NodeGraphSocket* meshInputSocket = context.node.input(NodeGraphValueType::Mesh);
    const EvaluatedSocketValue* inputMesh =
        meshInputSocket ? readEvaluatedInput(context.node, meshInputSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* upstreamGeometryValue = readInputValue(inputMesh);
    if (upstreamGeometryValue &&
        upstreamGeometryValue->dataType != payloadtypes::Geometry &&
        upstreamGeometryValue->dataType != payloadtypes::Remesh) {
        upstreamGeometryValue = nullptr;
    }

    const bool hasValidInput = payloadRegistry && upstreamGeometryValue;
    if (hasValidInput) {
        const RemeshNodeParams params = readRemeshNodeParams(context.node);
        remeshData.sourceMeshHandle = upstreamGeometryValue->payloadHandle;
        remeshData.iterations = params.iterations;
        remeshData.minAngleDegrees = static_cast<float>(params.minAngleDegrees);
        remeshData.maxEdgeLength = static_cast<float>(params.maxEdgeLength);
        remeshData.stepSize = static_cast<float>(params.stepSize);
        remeshData.active = true;
    }

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Remesh) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, remeshData, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeRemesh::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    const RemeshNodeParams params = readRemeshNodeParams(context.node);
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, nodegraphtypes::Remesh);
    HashNodeCache::combineSocket(hash, context, NodeGraphValueType::Mesh, HashDomain::Geometry);
    HashBuilder::combine(hash, static_cast<uint64_t>(params.iterations));
    HashBuilder::combineFloat(hash, static_cast<float>(params.minAngleDegrees));
    HashBuilder::combineFloat(hash, static_cast<float>(params.maxEdgeLength));
    HashBuilder::combineFloat(hash, static_cast<float>(params.stepSize));

    HashValues values{};
    values.full = hash;
    values.geometry = hash;
    values.simulation = hash;
    return values;
}
