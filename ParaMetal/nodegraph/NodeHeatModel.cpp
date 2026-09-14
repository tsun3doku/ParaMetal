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

void NodeHeatModel::execute(NodeKernelEval& eval) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(eval.node);
    const std::size_t meshSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Remesh);
    const NodeDataBlock* inputMeshValue =
        (meshSocketIndex < eval.inputs.size() && !eval.inputs[meshSocketIndex].empty())
            ? eval.inputs[meshSocketIndex].front() : nullptr;
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    NodeDataHandle meshHandle{};
    const bool hasValidInput = payloadRegistry && inputMeshValue && inputMeshValue->payloadHandle.key != 0;
    if (hasValidInput) {
        payloadRegistry->resolveRemesh(inputMeshValue->payloadHandle, &meshHandle);
    }

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
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
        payload.initialTemperatureC = static_cast<float>(params.initialTemperatureC);
        payload.volumetricHeatSource.powerDensity = static_cast<float>(params.volumetricPowerDensity);
        payload.prescribedTemperatureEnabled = params.prescribedTemperatureEnabled;
        payload.prescribedTemperatureC = static_cast<float>(params.prescribedTemperatureC);
        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeHeatModel::computeOutputHashes(const NodeKernelHash& hash) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(hash.node);
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::HeatModel);
    HashNodeCache::combineSocket(geometryHash, hash, NodeGraphValueType::Remesh, HashDomain::Geometry);

    uint64_t thermalHash = HashBuilder::start();
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.initialTemperatureC));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.volumetricPowerDensity));
    HashBuilder::combine(thermalHash, params.prescribedTemperatureEnabled ? 1u : 0u);
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.prescribedTemperatureC));

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
