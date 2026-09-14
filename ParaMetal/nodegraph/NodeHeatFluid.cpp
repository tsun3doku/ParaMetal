#include "NodeHeatFluid.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeHeatFluidParams.hpp"
#include "NodePayloadRegistry.hpp"

const char* NodeHeatFluid::typeId() const {
    return nodegraphtypes::HeatFluid;
}

void NodeHeatFluid::execute(NodeKernelEval& eval) const {
    const HeatFluidNodeParams params = readHeatFluidNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::HeatFluid) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        HeatFluidData payload{};
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperatureC = static_cast<float>(params.initialTemperatureC);
        payload.inletTemperatureC = static_cast<float>(params.inletTemperatureC);
        payload.velocity = params.velocity;

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeHeatFluid::computeOutputHashes(const NodeKernelHash& hash) const {
    const HeatFluidNodeParams params = readHeatFluidNodeParams(hash.node);
    uint64_t thermalHash = HashBuilder::start();
    HashBuilder::combineString(thermalHash, nodegraphtypes::HeatFluid);
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.initialTemperatureC));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.inletTemperatureC));
    HashBuilder::combineFloat(thermalHash, params.velocity.x);
    HashBuilder::combineFloat(thermalHash, params.velocity.y);
    HashBuilder::combineFloat(thermalHash, params.velocity.z);

    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, thermalHash);

    HashValues values{};
    values.full = simulationHash;
    values.thermal = thermalHash;
    values.simulation = simulationHash;
    return values;
}
