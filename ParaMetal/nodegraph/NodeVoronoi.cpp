#include "NodeVoronoi.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeVoronoiParams.hpp"
#include "domain/PointData.hpp"

#include <unordered_set>

const char* NodeVoronoi::typeId() const {
    return nodegraphtypes::Voronoi;
}

void NodeVoronoi::execute(NodeGraphKernelContext& context) const {
    NodeDataHandle modelMeshHandle;
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

    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Remesh);
    if (active && meshSocket) {
        const EvaluatedSocketValue* meshEval = readEvaluatedInput(context.node, meshSocket->id, context.executionState);
        const NodeDataBlock* meshData = readInputValue(meshEval);
        if (meshData && meshData->payloadHandle.key != 0) {
            const NodeDataHandle meshHandle = payloadRegistry ? payloadRegistry->resolveMeshHandle(meshData->dataType, meshData->payloadHandle) : NodeDataHandle{};
            if (meshHandle.key != 0) {
                modelMeshHandle = meshHandle;
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
        voronoiData.cellSize = static_cast<float>(nodeParams.sdfSize);
        voronoiData.voxelResolution = nodeParams.voxelResolution;
        voronoiData.domainType = domainType;
        voronoiData.modelMeshHandle = modelMeshHandle;
        voronoiData.pointsPayloadHandle = pointsPayloadHandle;
        voronoiData.active = active;
        const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[outputIndex].id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, voronoiData, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeVoronoi::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combineString(hash, nodegraphtypes::Voronoi);
    HashNodeCache::combineSocket(hash, context, NodeGraphValueType::Points, HashDomain::Geometry);
    HashNodeCache::combineOptionalSocket(hash, context, NodeGraphValueType::Remesh, HashDomain::Geometry);

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(context.node);
    HashBuilder::combineFloat(hash, static_cast<float>(nodeParams.sdfSize));
    HashBuilder::combine(hash, static_cast<uint64_t>(nodeParams.voxelResolution));

    HashValues values{};
    values.full = hash;
    values.geometry = hash;
    values.simulation = hash;
    return values;
}
