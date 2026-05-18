#include "NodeVoronoi.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphHash.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeVoronoiParams.hpp"

#include <iostream>
#include <unordered_set>

const char* NodeVoronoi::typeId() const {
    return nodegraphtypes::Voronoi;
}

void NodeVoronoi::execute(NodeGraphKernelContext& context) const {
    // Voronoi is a single-input node: it accepts exactly one mesh input.
    NodeDataHandle modelMeshHandle;
    uint64_t modelPayloadHash = 0;
    NodeDataHandle modelPayloadHandle;
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;

    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Mesh);
    if (meshSocket) {
        const auto evals = readEvaluatedInputs(context.node, meshSocket->id, context.executionState);
        for (const EvaluatedSocketValue* eval : evals) {
            if (!eval || eval->status != EvaluatedSocketStatus::Value) {
                continue;
            }

            const NodeDataBlock& inputValue = eval->data;
            if ((inputValue.dataType != payloadtypes::Geometry &&
                 inputValue.dataType != payloadtypes::Remesh &&
                 inputValue.dataType != payloadtypes::HeatModel) ||
                inputValue.payloadHandle.key == 0) {
                continue;
            }

            const NodeDataHandle meshHandle = payloadRegistry
                ? payloadRegistry->resolveMeshHandle(inputValue.dataType, inputValue.payloadHandle)
                : NodeDataHandle{};
            if (meshHandle.key == 0) {
                continue;
            }

            // Take the first valid mesh input only
            modelMeshHandle = meshHandle;
            modelPayloadHash = payloadRegistry->resolvePayloadHash(inputValue.payloadHandle);
            modelPayloadHandle = inputValue.payloadHandle;
            break;
        }
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(context.node);

    const bool active = modelMeshHandle.key != 0;
    for (std::size_t outputIndex = 0; outputIndex < context.outputs.size() && outputIndex < context.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = context.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = context.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Voronoi) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        VoronoiData voronoiData{};
        voronoiData.cellSize = static_cast<float>(nodeParams.cellSize);
        voronoiData.voxelResolution = nodeParams.voxelResolution;
        if (modelMeshHandle.key != 0) {
            voronoiData.modelMeshHandles.push_back(modelMeshHandle);
            voronoiData.modelPayloadHashes.push_back(modelPayloadHash);
            voronoiData.modelPayloadHandles.push_back(modelPayloadHandle);
        }
        voronoiData.active = active;
        const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[outputIndex].id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(voronoiData));
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeVoronoi::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));

    const NodeGraphSocket* meshSocket = context.node.input(NodeGraphValueType::Mesh);
    if (meshSocket) {
        const auto evals = readEvaluatedInputs(context.node, meshSocket->id, context.executionState);
        for (const EvaluatedSocketValue* input : evals) {
            const NodeDataBlock* inputData = nullptr;
            if (input && input->status == EvaluatedSocketStatus::Value) {
                if ((input->data.dataType == payloadtypes::Geometry ||
                     input->data.dataType == payloadtypes::Remesh ||
                     input->data.dataType == payloadtypes::HeatModel) &&
                    input->data.payloadHandle.key != 0) {
                    inputData = &input->data;
                }
            }
            NodeGraphHash::combineInputHash(outHash, inputData);
            // Only hash the first valid mesh input
            if (inputData) break;
        }
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(context.node);
    NodeGraphHash::combineFloat(outHash, static_cast<float>(nodeParams.cellSize));
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(nodeParams.voxelResolution));
    return true;
}
