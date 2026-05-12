#include "NodeHeatModel.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeGraphHash.hpp"
#include "NodeHeatModelParams.hpp"
#include "NodePayloadRegistry.hpp"

#include <iostream>

const char* NodeHeatModel::typeId() const {
    return nodegraphtypes::HeatModel;
}

void NodeHeatModel::execute(NodeGraphKernelContext& context) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(context.node);
    const NodeGraphSocket* meshSocket = findInputSocket(context.node, NodeGraphValueType::Mesh);
    const EvaluatedSocketValue* inputMesh =
        meshSocket ? readEvaluatedInput(context.node, meshSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* inputMeshValue = readInputValue(inputMesh);
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    NodeDataHandle meshHandle{};
    uint64_t meshPayloadHash = 0;
    const bool hasValidInput = payloadRegistry && inputMeshValue &&
        inputMeshValue->payloadHandle.key != 0 &&
        valueTypeOf(inputMeshValue->dataType) == NodeGraphValueType::Mesh;
    if (hasValidInput) {
        meshPayloadHash = payloadRegistry->resolvePayloadHash(inputMeshValue->dataType, inputMeshValue->payloadHandle);
        meshHandle = payloadRegistry->resolveMeshHandle(inputMeshValue->dataType, inputMeshValue->payloadHandle);
    }

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != NodePayloadType::HeatModel ||
            !hasValidInput || meshHandle.key == 0) {
            populateMetadata(outputValue, payloadRegistry);
            continue;
        }

        HeatModelData payload{};
        payload.meshHandle = meshHandle;
        payload.meshPayloadHash = meshPayloadHash;
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperature = static_cast<float>(params.initialTemperature);
        payload.boundaryCondition = params.boundaryCondition;
        payload.fixedTemperatureValue = static_cast<float>(params.fixedTemperatureValue);
        const uint64_t payloadKey = makeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload));
        populateMetadata(outputValue, payloadRegistry);
    }
}

bool NodeHeatModel::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(context.node);
    const NodeGraphSocket* meshSocket = findInputSocket(context.node, NodeGraphValueType::Mesh);
    const EvaluatedSocketValue* inputMesh =
        meshSocket ? readEvaluatedInput(context.node, meshSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* inputMeshValue = readInputValue(inputMesh);
    if (inputMeshValue && valueTypeOf(inputMeshValue->dataType) != NodeGraphValueType::Mesh) {
        inputMeshValue = nullptr;
    }

    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));
    NodeGraphHash::combineInputHash(outHash, inputMeshValue);
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.density));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.specificHeat));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.conductivity));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.initialTemperature));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(params.boundaryCondition));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.fixedTemperatureValue));
    return true;
}
