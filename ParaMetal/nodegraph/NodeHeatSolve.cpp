#include "NodeHeatSolve.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphHash.hpp"
#include "NodeHeatSolveParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "domain/HeatModelData.hpp"

#include <cstdint>
#include <iostream>
#include <set>
#include <vector>

const char* NodeHeatSolve::typeId() const {
    return nodegraphtypes::HeatSolve;
}

void NodeHeatSolve::execute(NodeGraphKernelContext& context) const {
    std::vector<NodeDataHandle> heatModelHandles;
    std::vector<NodeDataHandle> voronoiHandles;
    std::vector<NodeDataHandle> contactHandles;
    std::vector<HeatMaterialBinding> materialBindings;
    const HeatSolveNodeParams params = readHeatSolveNodeParams(context.node);

    const NodeGraphNodeId activeNodeId =
        selectHeatSolveNode(context.executionState.state, context.executionState);
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    // Collect all VoronoiData from Volume inputs (variadic socket)
    std::vector<const VoronoiData*> voronoiInputs;
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
                voronoiInputs.push_back(voronoi);
                voronoiHandles.push_back(block.payloadHandle);
            }
        }
    }

    // Collect all ContactData from Field inputs (variadic socket)
    std::vector<const ContactData*> contactInputs;
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
                contactInputs.push_back(contact);
                contactHandles.push_back(block.payloadHandle);
            }
        }
    }

    if (voronoiInputs.empty() || contactInputs.empty()) {
        populateOutputPayloads(
            context,
            heatModelHandles,
            voronoiHandles,
            contactHandles,
            materialBindings,
            0,
            0,
            16000.0f,
            false,
            false,
            false);
        return;
    }

    // Merge heat model handles from all Voronoi inputs
    std::set<NodeDataHandle> seenModelHandles;
    uint64_t combinedVoronoiHash = 0;
    for (const VoronoiData* voronoi : voronoiInputs) {
        combinedVoronoiHash ^= voronoi->payloadHash;
        if (voronoi->modelPayloadHandle.key != 0) {
            if (seenModelHandles.insert(voronoi->modelPayloadHandle).second) {
                heatModelHandles.push_back(voronoi->modelPayloadHandle);
            }
        }
    }

    // Also collect HeatModelData from direct Mesh inputs if any
    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Mesh);
    if (meshSocket) {
        const auto evals = readEvaluatedInputs(context.node, meshSocket->id, context.executionState);
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
                if (seenModelHandles.insert(heatModel->meshHandle).second) {
                    heatModelHandles.push_back(block.payloadHandle);
                }
            }
        }
    }

    materialBindings = params.materialBindings;

    // Combine all contact payload hashes
    uint64_t combinedContactHash = 0;
    for (const ContactData* contact : contactInputs) {
        combinedContactHash ^= contact->payloadHash;
    }

    const bool active = activeNodeId.isValid() && activeNodeId == context.node.id;
    const bool wantsPaused = params.resetRequested
        ? false
        : params.paused;
    populateOutputPayloads(
        context,
        heatModelHandles,
        voronoiHandles,
        contactHandles,
        materialBindings,
        combinedVoronoiHash,
        combinedContactHash,
        static_cast<float>(params.contactThermalConductance),
        active,
        active ? wantsPaused : false,
        active ? params.resetRequested : false);
}

void NodeHeatSolve::populateOutputPayloads(
    NodeGraphKernelContext& context,
    const std::vector<NodeDataHandle>& heatModelHandles,
    const std::vector<NodeDataHandle>& voronoiHandles,
    const std::vector<NodeDataHandle>& contactHandles,
    const std::vector<HeatMaterialBinding>& materialBindings,
    uint64_t voronoiPayloadHash,
    uint64_t contactPayloadHash,
    float contactThermalConductance,
    bool active,
    bool paused,
    bool resetRequested) {
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        const uint64_t payloadKey = NodeSocketKey(context.node.id, outputSocket.id);
        if (outputSocket.contract.producedPayloadType == payloadtypes::Heat) {
            HeatData heatData{};
            heatData.voronoiPayloadHash = voronoiPayloadHash;
            heatData.contactPayloadHash = contactPayloadHash;
            heatData.voronoiHandles = voronoiHandles;
            heatData.contactHandles = contactHandles;
            heatData.heatModelHandles = heatModelHandles;
            heatData.materialBindings = materialBindings;
            heatData.contactThermalConductance = contactThermalConductance;
            heatData.active = active;
            heatData.paused = paused;
            heatData.resetRequested = resetRequested;
            outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(heatData));
        }

        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeHeatSolve::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    const NodeGraphNodeId activeNodeId =
        selectHeatSolveNode(context.executionState.state, context.executionState);
    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));

    // Combine hashes for all Volume-type inputs containing Voronoi payloads (variadic socket)
    const NodeGraphSocket* voronoiSocket = context.node.input(NodeGraphValueType::Volume);
    if (voronoiSocket) {
        const auto voronoiEvals = readEvaluatedInputs(context.node, voronoiSocket->id, context.executionState);
        for (const EvaluatedSocketValue* input : voronoiEvals) {
            const NodeDataBlock* inputData = nullptr;
            if (input && input->status == EvaluatedSocketStatus::Value) {
                if (input->data.dataType == payloadtypes::Voronoi &&
                    input->data.payloadHandle.key != 0) {
                    inputData = &input->data;
                }
            }
            NodeGraphHash::combineInputHash(outHash, inputData);
        }
    }

    // Combine hashes for all Field-type inputs containing Contact payloads (variadic socket)
    const NodeGraphSocket* fieldSocketForHash = context.node.input(NodeGraphValueType::Field);
    if (fieldSocketForHash) {
        const auto contactEvals = readEvaluatedInputs(context.node, fieldSocketForHash->id, context.executionState);
        for (const EvaluatedSocketValue* input : contactEvals) {
            const NodeDataBlock* inputData = nullptr;
            if (input && input->status == EvaluatedSocketStatus::Value) {
                if (input->data.dataType == payloadtypes::Contact &&
                    input->data.payloadHandle.key != 0) {
                    inputData = &input->data;
                }
            }
            NodeGraphHash::combineInputHash(outHash, inputData);
        }
    }

    const HeatSolveNodeParams params = readHeatSolveNodeParams(context.node);
    const std::vector<HeatMaterialBinding>& materialBindings = params.materialBindings;
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(materialBindings.size()));
    for (const HeatMaterialBinding& binding : materialBindings) {
        NodeGraphHash::combine(outHash, static_cast<uint64_t>(binding.modelNodeId));
        NodeGraphHash::combine(outHash, static_cast<uint64_t>(binding.presetId));
    }
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.contactThermalConductance));
    const bool active = activeNodeId.isValid() && activeNodeId == context.node.id;
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(active ? 1u : 0u));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(active && params.paused ? 1u : 0u));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(active && params.resetRequested ? 1u : 0u));

    return true;
}

NodeGraphNodeId NodeHeatSolve::selectHeatSolveNode(
    const NodeGraphState& state,
    const NodeGraphKernelExecutionState& executionState) {
    (void)executionState;
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
