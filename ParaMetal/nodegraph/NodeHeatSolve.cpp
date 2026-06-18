#include "NodeHeatSolve.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "NodeHeatSolveParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "domain/ContactData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/VoronoiData.hpp"

#include <cstdint>
#include <vector>

const char* NodeHeatSolve::typeId() const {
    return nodegraphtypes::HeatSolve;
}

void NodeHeatSolve::execute(NodeGraphKernelContext& context) const {
    std::vector<NodeDataHandle> heatModelHandles;
    std::vector<NodeDataHandle> voronoiHandles;
    std::vector<NodeDataHandle> contactHandles;
    const HeatSolveNodeParams params = readHeatSolveNodeParams(context.node);

    const NodeGraphNodeId activeNodeId =
        selectHeatSolveNode(context.executionState.state);
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    // Collect active voronoi handles
    const NodeGraphSocket* voronoiSocket = context.node.input(NodeGraphValueType::Volume);
    if (voronoiSocket) {
        const auto evals = readEvaluatedInputs(context.node, voronoiSocket->id, context.executionState);
        for (const EvaluatedSocketValue* eval : evals) {
            if (!eval || eval->status != EvaluatedSocketStatus::Value) {
                continue;
            }
            const NodeDataBlock& block = eval->data;
            if (block.dataType != payloadtypes::Voronoi || block.payloadHandle.key == 0) {
                continue;
            }
            const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(block.payloadHandle);
            if (voronoi && voronoi->active) {
                voronoiHandles.push_back(block.payloadHandle);
            }
        }
    }

    // Collect active contact handles 
    const NodeGraphSocket* fieldSocket = context.node.input(NodeGraphValueType::Field);
    if (fieldSocket) {
        const auto evals = readEvaluatedInputs(context.node, fieldSocket->id, context.executionState);
        for (const EvaluatedSocketValue* eval : evals) {
            if (!eval || eval->status != EvaluatedSocketStatus::Value) {
                continue;
            }
            const NodeDataBlock& block = eval->data;
            if (block.dataType != payloadtypes::Contact || block.payloadHandle.key == 0) {
                continue;
            }
            const ContactData* contact = payloadRegistry->get<ContactData>(block.payloadHandle);
            if (contact && contact->active && contact->pair.hasValidContact) {
                contactHandles.push_back(block.payloadHandle);
            }
        }
    }

    // Collect heatmodel handles 
    if (!voronoiHandles.empty() && !contactHandles.empty()) {
        const NodeGraphSocket* heatModelSocket = context.node.input(NodeGraphValueType::HeatModel);
        if (heatModelSocket) {
            const auto evals = readEvaluatedInputs(context.node, heatModelSocket->id, context.executionState);
            for (const EvaluatedSocketValue* eval : evals) {
                if (!eval || eval->status != EvaluatedSocketStatus::Value) {
                    continue;
                }
                const NodeDataBlock& block = eval->data;
                if (block.dataType != payloadtypes::HeatModel || block.payloadHandle.key == 0) {
                    continue;
                }
                const HeatModelData* heatModel = payloadRegistry->get<HeatModelData>(block.payloadHandle);
                if (heatModel && heatModel->meshHandle.key != 0) {
                    heatModelHandles.push_back(block.payloadHandle);
                }
            }
        }
    }

    const bool active = activeNodeId.isValid() && activeNodeId == context.node.id;
    const bool hasValidInputs = !voronoiHandles.empty() && !contactHandles.empty();

    HeatData heatData{};
    heatData.voronoiHandles = voronoiHandles;
    heatData.contactHandles = contactHandles;
    heatData.heatModelHandles = heatModelHandles;
    heatData.contactThermalConductance = static_cast<float>(params.contactThermalConductance);
    heatData.simulationDuration = static_cast<float>(params.simulationDuration);
    heatData.active = hasValidInputs && active;
    heatData.paused = hasValidInputs && active && params.paused;
    heatData.resetCounter = hasValidInputs && active ? params.resetCounter : 0;
    heatData.rewindFrame = hasValidInputs && active ? params.rewindFrame : heat::NoRewindFrame;

    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputSocket.contract.producedPayloadType != payloadtypes::Heat) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, heatData, context.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodeHeatSolve::computeOutputHashes(const NodeGraphKernelHashContext& context) const {
    const NodeGraphNodeId activeNodeId =
        selectHeatSolveNode(context.executionState.state);
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combineString(simulationHash, nodegraphtypes::HeatSolve);
    HashNodeCache::combineSocketList(simulationHash, context, NodeGraphValueType::Volume, HashDomain::Simulation);
    HashNodeCache::combineSocketList(simulationHash, context, NodeGraphValueType::Field, HashDomain::Simulation);
    HashNodeCache::combineOptionalSocketList(simulationHash, context, NodeGraphValueType::HeatModel, HashDomain::Simulation);

    const HeatSolveNodeParams params = readHeatSolveNodeParams(context.node);
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.contactThermalConductance));
    HashBuilder::combineFloat(simulationHash, static_cast<float>(params.simulationDuration));
    const bool active = activeNodeId.isValid() && activeNodeId == context.node.id;
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(active ? 1u : 0u));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, static_cast<uint64_t>(active && params.paused ? 1u : 0u));
    HashBuilder::combine(fullHash, static_cast<uint64_t>(active ? params.resetCounter : 0u));
    HashBuilder::combine(fullHash, static_cast<uint64_t>(active ? params.rewindFrame : heat::NoRewindFrame));

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
