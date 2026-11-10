#include "NodeHeatProbe.hpp"

#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeHeatProbeParams.hpp"
#include "domain/HeatProbeData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"

const char* NodeHeatProbe::typeId() const { return nodegraphtypes::HeatProbe; }

void NodeHeatProbe::execute(NodeKernelEval& eval) const {
    if (eval.outputs.empty()) return;
    NodeDataBlock& output = eval.outputs.front();
    output = {};

    const NodeDataBlock* input = (!eval.inputs.empty() && !eval.inputs.front().empty())
        ? eval.inputs.front().front()
        : nullptr;

    if (input && input->dataType == payloadtypes::Heat) {
        output.payloadHandle = input->payloadHandle;
        output.dataType = payloadtypes::Heat;
    } else {
        output.dataType = payloadtypes::Heat;
    }
}

HashValues NodeHeatProbe::computeOutputHashes(const NodeKernelHash& hash) const {
    uint64_t upstreamSim = HashBuilder::start();
    HashNodeCache::combineSocket(upstreamSim, hash, NodeGraphValueType::Simulation, HashDomain::Simulation);

    const auto params = readHeatProbeNodeParams(hash.node);
    uint64_t fullValue = HashBuilder::start();
    HashBuilder::combineString(fullValue, nodegraphtypes::HeatProbe);
    HashBuilder::combine(fullValue, params.enabled ? 1u : 0u);
    HashBuilder::combine(fullValue, static_cast<uint64_t>(params.opcPort));
    HashBuilder::combine(fullValue, params.showTemperature ? 1u : 0u);
    HashBuilder::combine(fullValue, params.showName ? 1u : 0u);
    HashBuilder::combine(fullValue, static_cast<uint64_t>(params.items.size()));
    for (const auto& item : params.items) {
        HashBuilder::combine(fullValue, static_cast<uint64_t>(item.id));
        HashBuilder::combineString(fullValue, item.name);
        HashBuilder::combineFloat(fullValue, item.worldPosition.x);
        HashBuilder::combineFloat(fullValue, item.worldPosition.y);
        HashBuilder::combineFloat(fullValue, item.worldPosition.z);
        HashBuilder::combine(fullValue, item.enabled ? 1u : 0u);
    }
    HashBuilder::combine(fullValue, upstreamSim);

    HashValues hashes{};
    hashes.full = fullValue;
    hashes.thermal = fullValue;
    hashes.simulation = fullValue;
    return hashes;
}
