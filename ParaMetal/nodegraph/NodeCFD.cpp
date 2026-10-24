#include "NodeCFD.hpp"
#include "NodeCFDParams.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "domain/CFDData.hpp"
#include "domain/FluidData.hpp"
#include "domain/VoronoiData.hpp"

#include <cstdint>
#include <vector>

const char* NodeCFD::typeId() const {
    return nodegraphtypes::CFD;
}

void NodeCFD::execute(NodeKernelEval& eval) const {
    NodeDataHandle domainVoronoiHandle;
    const CFDNodeParams params = readCFDNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    const std::size_t voronoiSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Volume);
    if (voronoiSocketIndex < eval.inputs.size()) {
        for (const NodeDataBlock* block : eval.inputs[voronoiSocketIndex]) {
            if (!block || block->dataType != payloadtypes::Voronoi || block->payloadHandle.key == 0) {
                continue;
            }
            const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(block->payloadHandle);
            if (voronoi && voronoi->active) {
                if (domainVoronoiHandle.key == 0) {
                    domainVoronoiHandle = block->payloadHandle;
                }
            }
        }
    }

    NodeDataHandle fluidHandle{};
    bool hasFluid = false;
    const std::size_t fluidSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Fluid);
    if (fluidSocketIndex < eval.inputs.size() && !eval.inputs[fluidSocketIndex].empty()) {
        const NodeDataBlock* block = eval.inputs[fluidSocketIndex].front();
        if (block && block->dataType == payloadtypes::Fluid && block->payloadHandle.key != 0) {
            const FluidData* fluid = payloadRegistry->get<FluidData>(block->payloadHandle);
            if (fluid) {
                fluidHandle = block->payloadHandle;
                hasFluid = true;
            }
        }
    }

    const bool hasValidInputs = domainVoronoiHandle.key != 0 && hasFluid;
    const bool active = params.enabled && hasValidInputs;

    CFDData cfdData{};
    cfdData.domainVoronoiHandle = domainVoronoiHandle;
    cfdData.fluidHandle = fluidHandle;
    cfdData.hasFluid = hasFluid;
    cfdData.active = active;
    cfdData.fixedTimeStep = static_cast<float>(params.timeStep);
    cfdData.targetCourant = static_cast<float>(params.targetCourant);
    cfdData.pressureIterations = static_cast<int>(params.pressureIterations);
    cfdData.pressureTolerance = static_cast<float>(params.pressureTolerance);
    cfdData.initialVelocity = params.initialVelocity;
    cfdData.boundaries = params.boundaries;

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputSocket.contract.producedPayloadType != payloadtypes::CFD) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, cfdData, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeCFD::computeOutputHashes(const NodeKernelHash& hash) const {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combineString(simulationHash, nodegraphtypes::CFD);
    HashNodeCache::combineSocket(simulationHash, hash, NodeGraphValueType::Volume, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Fluid, HashDomain::Simulation);

    const CFDNodeParams params = readCFDNodeParams(hash.node);
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.timeStep));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.targetCourant));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.pressureIterations));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.pressureTolerance));
    HashBuilder::combinePod(simulationHash, params.initialVelocity);
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(params.enabled ? 1u : 0u));

    for (const auto& bc : params.boundaries) {
        HashBuilder::combine(
            simulationHash,
            static_cast<uint64_t>(bc.type));
        HashBuilder::combinePod(
            simulationHash,
            bc.velocity);
    }

    HashValues values{};
    values.full = simulationHash;
    values.simulation = simulationHash;
    return values;
}
