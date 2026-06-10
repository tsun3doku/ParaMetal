#include "NodeVoronoi.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphHash.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeVoronoiParams.hpp"
#include "domain/PointData.hpp"

#include <unordered_set>

const char* NodeVoronoi::typeId() const {
    return nodegraphtypes::Voronoi;
}

void NodeVoronoi::execute(NodeGraphKernelContext& context) const {
    NodeDataHandle modelMeshHandle;
    uint64_t modelPayloadHash = 0;
    NodeDataHandle modelPayloadHandle;
    NodeDataHandle pointsPayloadHandle;
    DomainType domainType = DomainType::Points;
    bool active = false;
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    const NodeGraphSocket* pointsSocket = context.node.input(NodeGraphValueType::Points);
    const EvaluatedSocketValue* pointsEval = pointsSocket ? readEvaluatedInput(context.node, pointsSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* pointsData = readInputValue(pointsEval);
    if (pointsData && pointsData->dataType == payloadtypes::Points && pointsData->payloadHandle.key != 0) {
        const PointData* pointData = payloadRegistry ? payloadRegistry->get<PointData>(pointsData->payloadHandle) : nullptr;
        if (pointData && !pointData->positions.empty()) {
            pointsPayloadHandle = pointsData->payloadHandle;
            active = true;
        }
    }

    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Mesh);
    if (active && meshSocket) {
        const EvaluatedSocketValue* meshEval = readEvaluatedInput(context.node, meshSocket->id, context.executionState);
        const NodeDataBlock* meshData = readInputValue(meshEval);
        if (meshData && meshData->payloadHandle.key != 0) {
            const NodeDataHandle meshHandle = payloadRegistry ? payloadRegistry->resolveMeshHandle(meshData->dataType, meshData->payloadHandle) : NodeDataHandle{};
            if (meshHandle.key != 0) {
                modelMeshHandle = meshHandle;
                modelPayloadHash = payloadRegistry->resolvePayloadHash(meshData->payloadHandle);
                modelPayloadHandle = meshData->payloadHandle;
                domainType = DomainType::Mesh;
            }
        }
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(context.node);

    for (std::size_t outputIndex = 0;
         outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Voronoi) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        VoronoiData voronoiData{};
        voronoiData.cellSize = static_cast<float>(nodeParams.cellSize);
        voronoiData.voxelResolution = nodeParams.voxelResolution;
        voronoiData.domainType = domainType;
        voronoiData.modelMeshHandle = modelMeshHandle;
        voronoiData.modelPayloadHash = modelPayloadHash;
        voronoiData.modelPayloadHandle = modelPayloadHandle;
        voronoiData.pointsPayloadHandle = pointsPayloadHandle;
        voronoiData.active = active;
        const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[outputIndex].id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(voronoiData));
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeVoronoi::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));

    const NodeGraphSocket* pointsSocket = context.node.input(NodeGraphValueType::Points);
    const EvaluatedSocketValue* pointsEval = pointsSocket ? readEvaluatedInput(context.node, pointsSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* pointsData = readInputValue(pointsEval);
    if (pointsData && pointsData->dataType == payloadtypes::Points && pointsData->payloadHandle.key != 0) {
        NodeGraphHash::combineInputHash(outHash, pointsData);
    } else {
        NodeGraphHash::combineInputHash(outHash, nullptr);
    }

    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Mesh);
    const EvaluatedSocketValue* meshEval = meshSocket ? readEvaluatedInput(context.node, meshSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* meshData = readInputValue(meshEval);
    if (meshData && meshData->payloadHandle.key != 0) {
        NodeGraphHash::combineInputHash(outHash, meshData);
    } else {
        NodeGraphHash::combineInputHash(outHash, nullptr);
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(context.node);
    NodeGraphHash::combineFloat(outHash, static_cast<float>(nodeParams.cellSize));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(nodeParams.voxelResolution));
    return true;
}
