#include "NodeHeatPoints.hpp"

#include "NodeGraphHash.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeHeatPointsParams.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/HeatPointsData.hpp"
#include "domain/PointData.hpp"

const char* NodeHeatPoints::typeId() const {
    return nodegraphtypes::HeatPoints;
}

void NodeHeatPoints::execute(NodeGraphKernelContext& context) const {
    const HeatPointsNodeParams params = readHeatPointsNodeParams(context.node);
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    // Resolve upstream PointData from the Points input socket
    NodeDataHandle upstreamPointsHandle;
    const PointData* upstreamPoints = nullptr;
    if (payloadRegistry) {
        const NodeGraphSocket* pointsSocket = context.node.input(NodeGraphValueType::Points);
        const EvaluatedSocketValue* pointsEval = pointsSocket ? readEvaluatedInput(context.node, pointsSocket->id, context.executionState) : nullptr;
        const NodeDataBlock* pointsData = readInputValue(pointsEval);
        if (pointsData && pointsData->dataType == payloadtypes::Points && pointsData->payloadHandle.key != 0) {
            upstreamPoints = payloadRegistry->get<PointData>(pointsData->payloadHandle);
            if (upstreamPoints && upstreamPoints->active && !upstreamPoints->positions.empty()) {
                upstreamPointsHandle = pointsData->payloadHandle;
            }
        }
    }

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::HeatPoints || !upstreamPoints) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        HeatPointsData payload{};
        payload.pointsHandle = upstreamPointsHandle;
        payload.initialTemperature = static_cast<float>(params.initialTemperature);
        payload.active = true;

        const std::size_t pointCount = upstreamPoints->positions.size();
        payload.boundaryConditions.reserve(pointCount);
        payload.fixedTemperatures.reserve(pointCount);
        const uint32_t bcValue = static_cast<uint32_t>(params.boundaryCondition);
        const float fixedTemp = static_cast<float>(params.fixedTemperature);
        for (std::size_t i = 0; i < pointCount; ++i) {
            payload.boundaryConditions.push_back(bcValue);
            payload.fixedTemperatures.push_back(fixedTemp);
        }
        payload.sealPayload();

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload));
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeHeatPoints::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    const HeatPointsNodeParams params = readHeatPointsNodeParams(context.node);
    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.initialTemperature));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.fixedTemperature));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(params.boundaryCondition));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(params.rows.size()));
    for (const HeatPointNodeRow& row : params.rows) {
        NodeGraphHash::combine(outHash, static_cast<uint64_t>(row.boundaryCondition));
        NodeGraphHash::combineFloat(outHash, row.fixedTemperature);
    }

    // Combine upstream PointData hash
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    if (payloadRegistry) {
        const NodeGraphSocket* pointsSocket = context.node.input(NodeGraphValueType::Points);
        const EvaluatedSocketValue* pointsEval = pointsSocket ? readEvaluatedInput(context.node, pointsSocket->id, context.executionState) : nullptr;
        const NodeDataBlock* pointsData = readInputValue(pointsEval);
        if (pointsData && pointsData->dataType == payloadtypes::Points && pointsData->payloadHandle.key != 0) {
            const uint64_t pointsHash = payloadRegistry->resolvePayloadHash(pointsData->payloadHandle);
            NodeGraphHash::combine(outHash, pointsHash);
        }
    }
    return true;
}
