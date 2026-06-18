#include "NodeMeshPoints.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/PointData.hpp"

const char* NodeMeshPoints::typeId() const {
    return nodegraphtypes::MeshPoints;
}

void NodeMeshPoints::execute(NodeGraphKernelContext& context) const {
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Points) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const NodeGraphSocket* inSocket = context.node.input("Geometry");
        const EvaluatedSocketValue* inputEval =
            inSocket ? readEvaluatedInput(context.node, inSocket->id, context.executionState) : nullptr;
        const NodeDataBlock* inputData = readInputValue(inputEval);

        if (!inputData || inputData->payloadHandle.key == 0) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const GeometryData* geometryData = payloadRegistry->resolveGeometry(inputData->payloadHandle);
        if (!geometryData || geometryData->pointPositions.empty()) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        PointData payload{};
        const std::size_t vertexCount = geometryData->pointPositions.size() / 3;
        payload.positions.reserve(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            payload.positions.push_back(glm::vec4(
                geometryData->pointPositions[3 * i + 0],
                geometryData->pointPositions[3 * i + 1],
                geometryData->pointPositions[3 * i + 2],
                1.0f
            ));
        }

        payload.localToWorld = geometryData->localToWorld;
        payload.active = true;
        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeMeshPoints::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, nodegraphtypes::MeshPoints);
    HashNodeCache::combineSocket(hash, context, "Geometry", HashDomain::Geometry);

    HashValues values{};
    values.full = hash;
    values.geometry = hash;
    values.simulation = hash;
    return values;
}
