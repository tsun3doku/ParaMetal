#include "NodeBoundaryConditions.hpp"
#include "NodeBoundaryConditionsParams.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"

#include "hash/HashBuilder.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

#include <cstdint>

const char* NodeBoundaryConditions::typeId() const {
    return nodegraphtypes::BoundaryConditions;
}

void NodeBoundaryConditions::execute(NodeKernelEval& eval) const {
    const BoundaryConditionsNodeParams params = readBoundaryConditionsNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Boundary) {
            continue;
        }

        BoundaryData payload{};
        payload.faces = params.faces;

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
    }
}

HashValues NodeBoundaryConditions::computeOutputHashes(const NodeKernelHash& hash) const {
    const BoundaryConditionsNodeParams params = readBoundaryConditionsNodeParams(hash.node);

    uint64_t boundaryHash = HashBuilder::start();
    HashBuilder::combineString(boundaryHash, nodegraphtypes::BoundaryConditions);
    for (const BoundaryData::Face& face : params.faces) {
        HashBuilder::combine(boundaryHash, static_cast<uint64_t>(face.flowType));
        HashBuilder::combinePod(boundaryHash, face.velocity);
        HashBuilder::combine(boundaryHash, static_cast<uint64_t>(face.thermalType));
        HashBuilder::combineFloat(boundaryHash, face.temperatureC);
        HashBuilder::combineFloat(boundaryHash, face.heatFlux);
        HashBuilder::combineFloat(boundaryHash, face.htc);
        HashBuilder::combineFloat(boundaryHash, face.inflowTemperatureC);
    }

    HashValues values{};
    values.full = boundaryHash;
    values.thermal = boundaryHash;
    values.simulation = boundaryHash;
    return values;
}
