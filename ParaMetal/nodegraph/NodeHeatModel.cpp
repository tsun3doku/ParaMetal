#include "NodeHeatModel.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeHeatModelParams.hpp"
#include "NodePayloadRegistry.hpp"

const char* NodeHeatModel::typeId() const {
    return nodegraphtypes::HeatModel;
}

void NodeHeatModel::execute(NodeGraphKernelContext& context) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(context.node);
    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Remesh);
    const EvaluatedSocketValue* inputMesh =
        meshSocket ? readEvaluatedInput(context.node, meshSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* inputMeshValue = readInputValue(inputMesh);
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    NodeDataHandle meshHandle{};
    const bool hasValidInput = payloadRegistry && inputMeshValue && inputMeshValue->payloadHandle.key != 0;
    if (hasValidInput) {
        meshHandle = payloadRegistry->resolveMeshHandle(inputMeshValue->dataType, inputMeshValue->payloadHandle);
    }

    for (std::size_t outputIndex = 0;
         outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::HeatModel || meshHandle.key == 0) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        HeatModelData payload{};
        payload.meshHandle = meshHandle;
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperature = static_cast<float>(params.initialTemperature);
        payload.boundaryCondition = params.boundaryCondition;
        payload.fixedTemperatureValue = static_cast<float>(params.fixedTemperatureValue);
        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeHeatModel::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(context.node);
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::HeatModel);
    HashNodeCache::combineSocket(geometryHash, context, NodeGraphValueType::Remesh, HashDomain::Geometry);

    uint64_t thermalHash = HashBuilder::start();
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.initialTemperature));
    HashBuilder::combine(thermalHash, static_cast<uint64_t>(params.boundaryCondition));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.fixedTemperatureValue));

    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, geometryHash);
    HashBuilder::combine(simulationHash, thermalHash);

    HashValues values{};
    values.full = simulationHash;
    values.geometry = geometryHash;
    values.thermal = thermalHash;
    values.simulation = simulationHash;
    return values;
}
