#include "NodeRemesh.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraph.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeRemeshParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

const char* NodeRemesh::typeId() const {
    return nodegraphtypes::Remesh;
}

void NodeRemesh::execute(NodeKernelEval& eval) const {
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;
    RemeshData remeshData{};

    const std::size_t meshSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Mesh);
    const NodeDataBlock* upstreamGeometryValue =
        (meshSocketIndex < eval.inputs.size() && !eval.inputs[meshSocketIndex].empty())
            ? eval.inputs[meshSocketIndex].front() : nullptr;
    if (upstreamGeometryValue &&
        upstreamGeometryValue->dataType != payloadtypes::Geometry &&
        upstreamGeometryValue->dataType != payloadtypes::Remesh) {
        upstreamGeometryValue = nullptr;
    }

    const bool hasValidInput = payloadRegistry && upstreamGeometryValue;
    if (hasValidInput) {
        const RemeshNodeParams params = readRemeshNodeParams(eval.node);
        remeshData.sourceMeshHandle = upstreamGeometryValue->payloadHandle;
        remeshData.iterations = params.iterations;
        remeshData.minAngleDegrees = static_cast<float>(params.minAngleDegrees);
        remeshData.maxEdgeLength = static_cast<float>(params.maxEdgeLength);
        remeshData.stepSize = static_cast<float>(params.stepSize);
        remeshData.active = true;
    }

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Remesh) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, remeshData, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeRemesh::computeOutputHashes(const NodeKernelHash& hash) const {
    const RemeshNodeParams params = readRemeshNodeParams(hash.node);
    uint64_t hashValue = HashBuilder::start();
    HashBuilder::combineString(hashValue, nodegraphtypes::Remesh);
    HashNodeCache::combineSocket(hashValue, hash, NodeGraphValueType::Mesh, HashDomain::Geometry);
    HashBuilder::combine(hashValue, static_cast<uint64_t>(params.iterations));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.minAngleDegrees));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.maxEdgeLength));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.stepSize));

    HashValues values{};
    values.full = hashValue;
    values.geometry = hashValue;
    values.simulation = hashValue;
    return values;
}
