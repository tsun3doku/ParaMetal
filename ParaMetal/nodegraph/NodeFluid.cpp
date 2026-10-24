#include "NodeFluid.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeFluidParams.hpp"
#include "NodePayloadRegistry.hpp"

const char* NodeFluid::typeId() const {
    return nodegraphtypes::Fluid;
}

void NodeFluid::execute(NodeKernelEval& eval) const {
    const FluidNodeParams params = readFluidNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Fluid) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        FluidData payload{};
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperatureC = static_cast<float>(params.initialTemperatureC);
        payload.kinematicViscosity = static_cast<float>(params.kinematicViscosity);

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeFluid::computeOutputHashes(const NodeKernelHash& hash) const {
    const FluidNodeParams params = readFluidNodeParams(hash.node);
    uint64_t thermalHash = HashBuilder::start();
    HashBuilder::combineString(thermalHash, nodegraphtypes::Fluid);
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.initialTemperatureC));
    HashBuilder::combineFloat(thermalHash, static_cast<float>(params.kinematicViscosity));

    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, thermalHash);

    HashValues values{};
    values.full = simulationHash;
    values.thermal = thermalHash;
    values.simulation = simulationHash;
    return values;
}
