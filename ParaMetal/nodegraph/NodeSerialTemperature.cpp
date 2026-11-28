#include "NodeSerialTemperature.hpp"

#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeSerialTemperatureParams.hpp"
#include "domain/SerialTemperatureData.hpp"
#include "hash/HashBuilder.hpp"

static NodeTypeDefinition buildSerialTemperatureNode() {
    return {
        nodegraphtypes::SerialTemperature,
        "Serial Temperature",
        NodeGraphNodeCategory::System,
        {
            makeOutputSocket(
                "Temperature", NodeGraphValueType::ScalarFloat, payloadtypes::SerialTemperature),
        },
        {
            {nodegraphparams::serialtemperature::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::serialtemperature::Port, "Port", NodeGraphParamType::String, 0.0, 0, false, "", false},
            {nodegraphparams::serialtemperature::BaudRate, "Baud Rate", NodeGraphParamType::Int, 0.0, 115200, false, "", false},
        },
    };
}

NodeTypeDefinition buildSerialTemperatureDefinition() {
    return buildSerialTemperatureNode();
}

std::unique_ptr<NodeKernel> createSerialTemperatureKernel() {
    return std::make_unique<NodeSerialTemperature>();
}

const char* NodeSerialTemperature::typeId() const { return nodegraphtypes::SerialTemperature; }

void NodeSerialTemperature::execute(NodeKernelEval& eval) const {
    if (eval.outputs.empty() || eval.node.outputs.empty()) return;
    NodeDataBlock& output = eval.outputs.front();
    output = {};
    output.dataType = eval.node.outputs.front().contract.producedPayloadType;
    if (!eval.runtime.payloadRegistry || output.dataType != payloadtypes::SerialTemperature) {
        return;
    }
    const auto params = readSerialTemperatureNodeParams(eval.node);
    SerialTemperatureData data{};
    data.enabled = params.enabled;
    data.portName = params.portName;
    data.baudRate = params.baudRate;
    const uint64_t key = NodeSocketKey(eval.node.id, eval.node.outputs.front().id);
    output.payloadHandle = eval.runtime.payloadRegistry->store(key, data, eval.outputHashes);
    if (output.payloadHandle.key == 0) {
        eval.error = "Payload publication backpressured.";
        return;
    }
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
    hashes.properties = value;
    hashes.simulation = value;
    return hashes;
}
