#include "NodeSerialTemperature.hpp"

#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeSerialTemperatureParams.hpp"
#include "domain/SerialTemperatureData.hpp"
#include "hash/HashBuilder.hpp"

const char* NodeSerialTemperature::typeId() const { return nodegraphtypes::SerialTemperature; }

void NodeSerialTemperature::execute(NodeKernelEval& eval) const {
    if (eval.outputs.empty() || eval.node.outputs.empty()) return;
    NodeDataBlock& output = eval.outputs.front();
    output = {};
    output.dataType = eval.node.outputs.front().contract.producedPayloadType;
    if (!eval.runtime.payloadRegistry || output.dataType != payloadtypes::SerialTemperature) {
        populateMetadata(output, nullptr, eval.runtime.payloadRegistry);
        return;
    }
    const auto params = readSerialTemperatureNodeParams(eval.node);
    SerialTemperatureData data{};
    data.enabled = params.enabled;
    data.portName = params.portName;
    data.baudRate = params.baudRate;
    const uint64_t key = NodeSocketKey(eval.node.id, eval.node.outputs.front().id);
    output.payloadHandle = eval.runtime.payloadRegistry->store(key, data, eval.outputHashes);
    populateMetadata(output, nullptr, eval.runtime.payloadRegistry);
}

HashValues NodeSerialTemperature::computeOutputHashes(const NodeKernelHash& hash) const {
    const auto params = readSerialTemperatureNodeParams(hash.node);
    uint64_t value = HashBuilder::start();
    HashBuilder::combineString(value, nodegraphtypes::SerialTemperature);
    HashBuilder::combine(value, params.enabled ? 1u : 0u);
    HashBuilder::combineString(value, params.portName);
    HashBuilder::combine(value, params.baudRate);
    HashValues hashes{};
    hashes.full = value;
    hashes.thermal = value;
    hashes.simulation = value;
    return hashes;
}
