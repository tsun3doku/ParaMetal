#include "NodeHeatSolve.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "NodeHeatSolveParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "domain/BoundaryData.hpp"
#include "domain/FluidData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/VoronoiData.hpp"

#include <cstdint>
#include <vector>

static NodeTypeDefinition buildHeatSolveNode() {
    return {
        nodegraphtypes::HeatSolve,
        "Ember",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Volume", NodeGraphValueType::Volume),
            makeInputSocket("HeatModel", NodeGraphValueType::HeatModel, true, false),
            makeInputSocket("Fluid", NodeGraphValueType::Fluid, false, false),
            makeInputSocket("CFD", NodeGraphValueType::Simulation, false, false),
            makeInputSocket("Boundary", NodeGraphValueType::Boundary),
            makeOutputSocket("Heat", NodeGraphValueType::Simulation, payloadtypes::Heat, true),
        },
        {
            {nodegraphparams::heatsolve::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::Paused, "Paused", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ResetRequested, "Reset Requested", NodeGraphParamType::Int, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowHeatOverlay, "Show Heat Overlay", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::ShowFluxVectors, "Show Flux Vectors", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowHeatPalette, "Show Heat Palette", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowIsocontours, "Show Isocontours", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::IsocontourSpacingC, "Isocontour Spacing (C)", NodeGraphParamType::Float, 10.0, 0, false, "", false},
            {nodegraphparams::heatsolve::FluxVectorScale, "Flux Vector Scale", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::heatsolve::RewindFrame, "Rewind Frame", NodeGraphParamType::Int, 0.0, static_cast<int64_t>(heat::NoRewindFrame), false, "", false},
            {nodegraphparams::heatsolve::SimulationDuration, "Simulation Duration", NodeGraphParamType::Float, 10.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowDomainBox, "Show Domain Box", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::ShowFluidVolume, "Show Fluid Volume", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::FluidVolumeOpacity, "Fluid Volume Opacity", NodeGraphParamType::Float, 0.25, 0, false, "", false},
            {nodegraphparams::heatsolve::SolverMaxIterations, "Max PCG Iterations (0=default)", NodeGraphParamType::Int, 100.0, 100, false, "", false},
            {nodegraphparams::heatsolve::SolverConvergenceTolerance, "PCG Tolerance (0=default)", NodeGraphParamType::Float, 3e-3, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildHeatSolveDefinition() {
    return buildHeatSolveNode();
}

std::unique_ptr<NodeKernel> createHeatSolveKernel() {
    return std::make_unique<NodeHeatSolve>();
}

const char* NodeHeatSolve::typeId() const {
    return nodegraphtypes::HeatSolve;
}

void NodeHeatSolve::execute(NodeKernelEval& eval) const {
    NodeDataHandle domainVoronoiHandle;
    std::vector<NodeDataHandle> heatModelHandles;
    uint32_t activeVoronoiCount = 0;
    const HeatSolveNodeParams params = readHeatSolveNodeParams(eval.node);

    const NodeGraphNodeId activeNodeId = selectHeatSolveNode(eval.runtime.graph);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    const std::size_t voronoiSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Volume);
    if (voronoiSocketIndex < eval.inputs.size()) {
        for (const NodeDataBlock* block : eval.inputs[voronoiSocketIndex]) {
            if (!block || block->dataType != payloadtypes::Voronoi || block->payloadHandle.key == 0) {
                continue;
            }
            const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(block->payloadHandle);
            if (voronoi && voronoi->active) {
                ++activeVoronoiCount;
                if (domainVoronoiHandle.key == 0) {
                    domainVoronoiHandle = block->payloadHandle;
                }
            }
        }
    }

    const std::size_t heatModelSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::HeatModel);
    if (heatModelSocketIndex < eval.inputs.size()) {
        for (const NodeDataBlock* block : eval.inputs[heatModelSocketIndex]) {
            if (!block || block->dataType != payloadtypes::HeatModel || block->payloadHandle.key == 0) {
                continue;
            }
            const HeatModelData* heatModel = payloadRegistry->get<HeatModelData>(block->payloadHandle);
            if (heatModel && heatModel->meshHandle.key != 0) {
                heatModelHandles.push_back(block->payloadHandle);
            }
        }
    }

    NodeDataHandle heatFluidHandle{};
    bool hasFluid = false;
    const std::size_t heatFluidSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Fluid);
    if (heatFluidSocketIndex < eval.inputs.size() && !eval.inputs[heatFluidSocketIndex].empty()) {
        const NodeDataBlock* block = eval.inputs[heatFluidSocketIndex].front();
        if (block && block->dataType == payloadtypes::Fluid && block->payloadHandle.key != 0) {
            const FluidData* fluid = payloadRegistry->get<FluidData>(block->payloadHandle);
            if (fluid) {
                heatFluidHandle = block->payloadHandle;
                hasFluid = true;
            }
        }
    }

    NodeDataHandle cfdHandle{};
    const std::size_t cfdSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Simulation);
    if (cfdSocketIndex < eval.inputs.size() && !eval.inputs[cfdSocketIndex].empty()) {
        const NodeDataBlock* block = eval.inputs[cfdSocketIndex].front();
        if (block && block->dataType == payloadtypes::CFD && block->payloadHandle.key != 0) {
            cfdHandle = block->payloadHandle;
        }
    }

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

    const bool active = activeNodeId.isValid() && activeNodeId == eval.node.id;
    const bool hasValidInputs = domainVoronoiHandle.key != 0 && !heatModelHandles.empty();

    HeatData heatData{};
    heatData.domainVoronoiHandle = domainVoronoiHandle;
    heatData.heatModelHandles = heatModelHandles;
    heatData.heatFluidHandle = heatFluidHandle;
    heatData.cfdHandle = cfdHandle;
    heatData.boundaryHandle = boundaryHandle;
    heatData.activeVoronoiCount = activeVoronoiCount;
    heatData.simulationDuration = static_cast<float>(params.simulationDuration);
    heatData.solverMaxIterations = params.solverMaxIterations;
    heatData.solverConvergenceTolerance = params.solverConvergenceTolerance;
    heatData.isocontourSpacingC = params.isocontourSpacingC;
    heatData.hasFluid = hasFluid;
    heatData.active = hasValidInputs && active;

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputSocket.contract.producedPayloadType != payloadtypes::Heat) {
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, heatData, eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeHeatSolve::computeOutputHashes(const NodeKernelHash& hash) const {
    const NodeGraphNodeId activeNodeId = selectHeatSolveNode(hash.runtime.graph);
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combineString(simulationHash, nodegraphtypes::HeatSolve);
    HashNodeCache::combineSocket(simulationHash, hash, NodeGraphValueType::Volume, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocketList(simulationHash, hash, NodeGraphValueType::HeatModel, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Fluid, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Simulation, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocket(simulationHash, hash, NodeGraphValueType::Boundary, HashDomain::Simulation);

    const HeatSolveNodeParams params = readHeatSolveNodeParams(hash.node);
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.simulationDuration));
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(params.solverMaxIterations));
    HashBuilder::combineFloat(simulationHash, params.solverConvergenceTolerance);
    const bool active = activeNodeId.isValid() && activeNodeId == hash.node.id;
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(active ? 1u : 0u));

    uint64_t fullHash = simulationHash;
    HashBuilder::combineFloat(fullHash, params.isocontourSpacingC);

    HashValues values{};
    values.full = fullHash;
    values.simulation = simulationHash;
    return values;
}

NodeGraphNodeId NodeHeatSolve::selectHeatSolveNode(
    const NodeGraphState& state) {
    NodeGraphNodeId selectedNodeId{};
    for (const auto& [id, node] : state.nodes) {
        if (getNodeTypeId(node.typeId) != nodegraphtypes::HeatSolve) {
            continue;
        }

        const HeatSolveNodeParams params = readHeatSolveNodeParams(node);
        if (!params.enabled) {
            continue;
        }

        if (selectedNodeId.isValid()) {
            return {};
        }

        selectedNodeId = node.id;
    }

    return selectedNodeId;
}
