#include "NodeMeshPoints.hpp"

#include "NodeGraphDefinitionUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/PointData.hpp"

static NodeTypeDefinition buildMeshPointsNode() {
    return {
        nodegraphtypes::MeshPoints,
        "Mesh Points",
        NodeGraphNodeCategory::Geometry,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Remesh}),
            makeOutputSocket("Points", NodeGraphValueType::Points, payloadtypes::Points, true),
        },
        {},
    };
}

NodeTypeDefinition buildMeshPointsDefinition() {
    return buildMeshPointsNode();
}

std::unique_ptr<NodeKernel> createMeshPointsKernel() {
    return std::make_unique<NodeMeshPoints>();
}

const char* NodeMeshPoints::typeId() const {
    return nodegraphtypes::MeshPoints;
}

void NodeMeshPoints::execute(NodeKernelEval& eval) const {
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;
    const std::size_t geometrySocketIndex = inputIndexOf(eval.node, "Geometry");
    const NodeDataBlock* inputData =
        (geometrySocketIndex < eval.inputs.size() && !eval.inputs[geometrySocketIndex].empty())
            ? eval.inputs[geometrySocketIndex].front() : nullptr;

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Points) {
            continue;
        }

        if (!inputData || inputData->payloadHandle.key == 0) {
            continue;
        }

        const bool isRemesh = payloadRegistry->resolveRemesh(inputData->payloadHandle) != nullptr;
        const GeometryData* geometryData = isRemesh
            ? nullptr
            : payloadRegistry->resolveGeometry(inputData->payloadHandle);
        if (!isRemesh && !geometryData) {
            eval.error = "Mesh Points input is not Geometry or Remesh data.";
            return;
        }

        PointData payload{};
        payload.sources.push_back(inputData->payloadHandle);
        if (!payloadRegistry->resolveLocalToWorld(
                inputData->payloadHandle,
                payload.localToWorld)) {
            eval.error = "Mesh Points cannot resolve its source placement.";
            return;
        }
        payload.active = true;
        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeMeshPoints::computeOutputHashes(const NodeKernelHash& hash) const {
    uint64_t hashValue = HashBuilder::start();
    HashBuilder::combineString(hashValue, nodegraphtypes::MeshPoints);
    HashNodeCache::combineSocket(hashValue, hash, "Geometry", HashDomain::Geometry);

    HashValues values{};
    values.full = hashValue;
    values.geometry = hashValue;
    values.simulation = hashValue;
    return values;
}
