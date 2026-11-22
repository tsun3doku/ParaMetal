#include "NodeCFD.hpp"
#include "NodeCFDParams.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "domain/BoundaryData.hpp"
#include "domain/CFDData.hpp"
#include "domain/FluidData.hpp"
#include "domain/VoronoiData.hpp"

#include <cstdint>
#include <vector>

static NodeTypeDefinition buildCFDNode() {
    return {
        nodegraphtypes::CFD,
        "Superfluid",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Volume", NodeGraphValueType::Volume, false, true),
            makeInputSocket("Fluid", NodeGraphValueType::Fluid, false, false),
            makeInputSocket("Boundary", NodeGraphValueType::Boundary),
            makeOutputSocket("CFD", NodeGraphValueType::Simulation, payloadtypes::CFD, true),
        },
        {
            {nodegraphparams::cfd::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::cfd::TargetCourant, "Target Courant", NodeGraphParamType::Float, 0.5, 0, false, "", false},
            {nodegraphparams::cfd::TimeStep, "Time Step (s)", NodeGraphParamType::Float, 1.0 / 60.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityX, "Initial Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityY, "Initial Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityZ, "Initial Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::PressureIterations, "Pressure Iterations", NodeGraphParamType::Int, 200.0, 200, false, "", false},
            {nodegraphparams::cfd::PressureTolerance, "Pressure Tolerance", NodeGraphParamType::Float, 1e-4, 0, false, "", false},
            {nodegraphparams::cfd::Duration, "Simulation Duration (s)", NodeGraphParamType::Float, 10.0, 0, false, "", false},

            {nodegraphparams::cfd::ShowStreamlines, "Show Streamlines", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::cfd::ShowDomainBox, "Show Domain Box", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::cfd::ShowVelocityPalette, "Show Velocity Palette", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::ShowSlice, "Show Slice", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            makeEnumParamDefinition(nodegraphparams::cfd::SliceAxis, "Slice Axis", "X", {"X", "Y", "Z"}),
            {nodegraphparams::cfd::SlicePosition, "Slice Position", NodeGraphParamType::Float, 0.5, 0, false, "", false},
            {nodegraphparams::cfd::ShowSliceWireframe, "Show Slice Wireframe", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildCFDDefinition() {
    return buildCFDNode();
}

std::unique_ptr<NodeKernel> createCFDKernel() {
    return std::make_unique<NodeCFD>();
}

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

    NodeDataHandle boundaryHandle{};
    const std::size_t boundarySocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Boundary);
    if (boundarySocketIndex < eval.inputs.size() && !eval.inputs[boundarySocketIndex].empty()) {
        const NodeDataBlock* block = eval.inputs[boundarySocketIndex].front();
        if (block && block->dataType == payloadtypes::Boundary && block->payloadHandle.key != 0) {
            const BoundaryData* boundary = payloadRegistry->get<BoundaryData>(block->payloadHandle);
            if (boundary) {
                boundaryHandle = block->payloadHandle;
            }
        }
    }

    CFDData cfdData{};
    cfdData.domainVoronoiHandle = domainVoronoiHandle;
    cfdData.fluidHandle = fluidHandle;
    cfdData.boundaryHandle = boundaryHandle;
    cfdData.hasFluid = hasFluid;
    cfdData.active = active;
    cfdData.fixedTimeStep = static_cast<float>(params.timeStep);
    cfdData.targetCourant = static_cast<float>(params.targetCourant);
    cfdData.pressureIterations = static_cast<int>(params.pressureIterations);
    cfdData.pressureTolerance = static_cast<float>(params.pressureTolerance);
    cfdData.simulationDuration = static_cast<float>(params.simulationDuration);
    cfdData.initialVelocity = params.initialVelocity;

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputSocket.contract.producedPayloadType != payloadtypes::CFD) {
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, cfdData, eval.outputHashes);
    }
}

HashValues NodeCFD::computeOutputHashes(const NodeKernelHash& hash) const {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combineString(simulationHash, nodegraphtypes::CFD);
    HashNodeCache::combineSocket(simulationHash, hash, NodeGraphValueType::Volume, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Fluid, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Boundary, HashDomain::Simulation);

    const CFDNodeParams params = readCFDNodeParams(hash.node);
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.timeStep));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.targetCourant));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.pressureIterations));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.pressureTolerance));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.simulationDuration));
    HashBuilder::combinePod(simulationHash, params.initialVelocity);
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(params.enabled ? 1u : 0u));

    HashValues values{};
    values.full = simulationHash;
    values.simulation = simulationHash;
    return values;
}
