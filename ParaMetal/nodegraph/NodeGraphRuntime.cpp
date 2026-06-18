#include "NodeGraphRuntime.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphBridge.hpp"
#include "NodePayloadRegistry.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

NodeGraphRuntime::NodeGraphRuntime(NodeGraphBridge* nodeGraphBridge, const NodeRuntimeServices& services)
    : bridge(nodeGraphBridge),
      runtimeServices(services) {
}

NodeGraphRuntime::~NodeGraphRuntime() = default;

EvaluatedSocketValue NodeGraphRuntime::makeMissingSocketValue() const {
    EvaluatedSocketValue value{};
    value.status = EvaluatedSocketStatus::Missing;
    return value;
}

EvaluatedSocketValue NodeGraphRuntime::makeErrorSocketValue(std::string error) const {
    EvaluatedSocketValue value{};
    value.status = EvaluatedSocketStatus::Error;
    value.error = std::move(error);
    return value;
}

EvaluatedSocketValue NodeGraphRuntime::makeValueSocketValue(const NodeDataBlock& data) const {
    EvaluatedSocketValue value{};
    value.status = EvaluatedSocketStatus::Value;
    value.data = data;
    return value;
}

void NodeGraphRuntime::propagateSkippedNodeOutputs(
    const NodeGraphNode& node,
    EvaluatedSocketStatus status,
    const std::string& error,
    NodeGraphEvaluationState& state) const {
    const EvaluatedSocketValue skippedValue =
        (status == EvaluatedSocketStatus::Error)
        ? makeErrorSocketValue(error)
        : makeMissingSocketValue();
    for (const NodeGraphSocket& outputSocket : node.outputs) {
        state.outputBySocket[NodeSocketKey(node.id, outputSocket.id).value] = skippedValue;
    }
}

bool NodeGraphRuntime::evaluateNodeInputs(
    const NodeGraphNode& node,
    const std::unordered_map<uint64_t, const NodeGraphEdge*>& incomingEdgeByInputSocket,
    const NodeGraphEvaluationState& state,
    std::vector<const EvaluatedSocketValue*>& outInputs,
    EvaluatedSocketStatus& outStatus,
    std::string& outError) const {
    outInputs.assign(node.inputs.size(), nullptr);
    outStatus = EvaluatedSocketStatus::Value;
    outError.clear();

    for (std::size_t inputIndex = 0; inputIndex < node.inputs.size(); ++inputIndex) {
        const NodeGraphSocket& inputSocket = node.inputs[inputIndex];
        const auto edgeIt = incomingEdgeByInputSocket.find(NodeSocketKey(node.id, inputSocket.id).value);
        if (edgeIt == incomingEdgeByInputSocket.end() || !edgeIt->second) {
            if (!inputSocket.required) {
                outInputs[inputIndex] = nullptr;
                continue;
            }
            outStatus = EvaluatedSocketStatus::Missing;
            return false;
        }

        const NodeGraphEdge& edge = *edgeIt->second;
        const auto outputIt = state.outputBySocket.find(NodeSocketKey(edge.fromNode, edge.fromSocket).value);
        if (outputIt == state.outputBySocket.end()) {
            if (!inputSocket.required) {
                outInputs[inputIndex] = nullptr;
                continue;
            }
            outStatus = EvaluatedSocketStatus::Missing;
            return false;
        }

        const EvaluatedSocketValue& inputValue = outputIt->second;
        outInputs[inputIndex] = &inputValue;
        if (inputValue.status == EvaluatedSocketStatus::Value) {
            continue;
        }

        outStatus = inputValue.status;
        outError = inputValue.error;
        return false;
    }

    return true;
}

void NodeGraphRuntime::applyDelta(const NodeGraphDelta& delta) {
    if (!delta.changes.empty()) {
        bool shouldClearCaches = false;
        std::unordered_set<uint32_t> dirtyNodeIds;
        for (const NodeGraphChange& change : delta.changes) {
            if (change.reason == NodeGraphChangeReason::Topology) {
                shouldClearCaches = true;
            }
            if (change.reason == NodeGraphChangeReason::Parameter &&
                change.type == NodeGraphChangeType::NodeUpsert &&
                change.node.id.isValid()) {
                dirtyNodeIds.insert(change.node.id.value);
            }
            applyChange(change);
        }
        if (shouldClearCaches) {
            clearNodeCaches();
            if (runtimeServices.payloadRegistry) {
                runtimeServices.payloadRegistry->clear();
            }
        } else if (!dirtyNodeIds.empty()) {
            for (uint32_t nodeId : dirtyNodeIds) {
                lastHashByNodeId.erase(nodeId);
                cachedOutputsByNodeId.erase(nodeId);
            }
        }
    }
    graphState.revision = delta.toRevision;
}

void NodeGraphRuntime::applyChange(const NodeGraphChange& change) {
    switch (change.type) {
    case NodeGraphChangeType::Reset:
        graphState.nodes.clear();
        graphState.edges.clear();
        break;
    case NodeGraphChangeType::NodeUpsert: {
        graphState.nodes[change.node.id.value] = change.node;
        break;
    }
    case NodeGraphChangeType::NodeRemoved:
        graphState.nodes.erase(change.nodeId.value);
        for (auto it = graphState.edges.begin(); it != graphState.edges.end(); ) {
            if (it->second.fromNode == change.nodeId || it->second.toNode == change.nodeId) {
                it = graphState.edges.erase(it);
            } else {
                ++it;
            }
        }
        break;
    case NodeGraphChangeType::EdgeUpsert:
        graphState.edges[change.edge.id.value] = change.edge;
        break;
    case NodeGraphChangeType::EdgeRemoved:
        graphState.edges.erase(change.edgeId.value);
        break;
    }
}

void NodeGraphRuntime::tick(NodeGraphEvaluationState* outState, const NodeGraphCompiled& compiled) {
    if (!bridge) {
        if (outState) {
            outState->upstreamSocket.clear();
            outState->outputBySocket.clear();
        }
        return;
    }
    execute(outState, compiled);
}

void NodeGraphRuntime::execute(NodeGraphEvaluationState* outState, const NodeGraphCompiled& compiled) {
    if (graphState.nodes.size() > 0 && compiled.executionOrder.size() != graphState.nodes.size()) {
        if (outState) {
            outState->upstreamSocket.clear();
            outState->outputBySocket.clear();
        }
        return;
    }

    const std::vector<NodeGraphNodeId>& executionOrder = compiled.executionOrder;
    std::unordered_map<uint64_t, const NodeGraphEdge*> incomingEdgeByInputSocket;
    incomingEdgeByInputSocket.reserve(graphState.edges.size() * 2);
    std::unordered_map<uint64_t, std::vector<const NodeGraphEdge*>> incomingEdgesByInputSocket;
    incomingEdgesByInputSocket.reserve(graphState.edges.size() * 2);
    NodeGraphEvaluationState state{};
    state.upstreamSocket.reserve(graphState.edges.size() * 2);
    state.outputBySocket.reserve(graphState.edges.size() * 2);
    for (const auto& [id, edge] : graphState.edges) {
        const uint64_t inputKey = NodeSocketKey(edge.toNode, edge.toSocket).value;
        incomingEdgeByInputSocket[inputKey] = &edge;
        incomingEdgesByInputSocket[inputKey].push_back(&edge);
        const uint64_t sourceKey = NodeSocketKey(edge.fromNode, edge.fromSocket).value;
        state.upstreamSocket[inputKey] = sourceKey;
        state.upstreamSockets[inputKey].push_back(sourceKey);
    }

    for (NodeGraphNodeId nodeId : executionOrder) {
        auto it = graphState.nodes.find(nodeId.value);
        if (it == graphState.nodes.end()) {
            continue;
        }

        const NodeGraphNode& node = it->second;
        const NodeTypeId typeId = getNodeTypeId(node.typeId);

        std::vector<const EvaluatedSocketValue*> inputValues;
        inputValues.reserve(node.inputs.size());
        EvaluatedSocketStatus blockedStatus = EvaluatedSocketStatus::Value;
        std::string blockedError;
        if (!evaluateNodeInputs(
                node,
                incomingEdgeByInputSocket,
                state,
                inputValues,
                blockedStatus,
                blockedError)) {
            propagateSkippedNodeOutputs(node, blockedStatus, blockedError, state);
            continue;
        }

        std::vector<NodeDataBlock> outputValues;
        outputValues.reserve(node.outputs.size());
        if (kernels.hasKernel(typeId)) {
            const NodeGraphKernelExecutionState kernelState{
                graphState,
                *bridge,
                runtimeServices,
                incomingEdgeByInputSocket,
                incomingEdgesByInputSocket,
                state.outputBySocket};

            HashValues outputHashes = kernels.computeOutputHashes(node, kernelState, inputValues);
            uint64_t cacheHash = outputHashes.full;
            bool reusedCache = false;
            const auto hashIt = lastHashByNodeId.find(node.id.value);
            const auto cacheIt = cachedOutputsByNodeId.find(node.id.value);
            if (hashIt != lastHashByNodeId.end() &&
                cacheIt != cachedOutputsByNodeId.end() &&
                hashIt->second == cacheHash &&
                cacheIt->second.size() == node.outputs.size()) {
                outputValues = cacheIt->second;
                reusedCache = true;
            }

            if (!reusedCache) {
                outputValues.resize(node.outputs.size());
                std::vector<const NodeDataBlock*> inputDataValues(inputValues.size(), nullptr);
                for (std::size_t inputIndex = 0; inputIndex < inputValues.size(); ++inputIndex) {
                    const EvaluatedSocketValue* inputValue = inputValues[inputIndex];
                    if (inputValue && inputValue->status == EvaluatedSocketStatus::Value) {
                        inputDataValues[inputIndex] = &inputValue->data;
                    }
                }
                {
                    std::unordered_map<std::string, std::string> mergedMetadata;
                    std::vector<NodeGraphNodeId> mergedLineage;
                    std::unordered_set<uint32_t> seenNodeIds;
                    for (const NodeDataBlock* input : inputDataValues) {
                        if (!input) continue;
                        for (const auto& metadataEntry : input->metadata) {
                            mergedMetadata[metadataEntry.first] = metadataEntry.second;
                        }
                        for (NodeGraphNodeId lineageNodeId : input->lineageNodeIds) {
                            if (!lineageNodeId.isValid()) continue;
                            if (seenNodeIds.insert(lineageNodeId.value).second) {
                                mergedLineage.push_back(lineageNodeId);
                            }
                        }
                    }
                    if (node.id.isValid() && seenNodeIds.insert(node.id.value).second) {
                        mergedLineage.push_back(node.id);
                    }
                    mergedMetadata["graph.producer_node_id"] = std::to_string(node.id.value);
                    mergedMetadata["graph.producer_type_id"] = getNodeTypeId(node.typeId);
                    mergedMetadata["graph.lineage_depth"] = std::to_string(mergedLineage.size());
                    for (NodeDataBlock& output : outputValues) {
                        output = {};
                        output.metadata = mergedMetadata;
                        output.lineageNodeIds = mergedLineage;
                        populateMetadata(output, nullptr, nullptr);
                    }
                }
                kernels.executeNode(node, kernelState, inputValues, outputValues, outputHashes);

                lastHashByNodeId[node.id.value] = cacheHash;
                cachedOutputsByNodeId[node.id.value] = outputValues;
            }

            for (std::size_t outputIndex = 0; outputIndex < node.outputs.size(); ++outputIndex) {
                state.outputBySocket[NodeSocketKey(node.id, node.outputs[outputIndex].id).value] =
                    makeValueSocketValue(outputValues[outputIndex]);
            }
        }
    }

    if (outState) {
        outState->upstreamSocket = std::move(state.upstreamSocket);
        outState->upstreamSockets = std::move(state.upstreamSockets);
        outState->outputBySocket = std::move(state.outputBySocket);
        outState->executionOrder = executionOrder;
    }
}

void NodeGraphRuntime::clearNodeCaches() {
    lastHashByNodeId.clear();
    cachedOutputsByNodeId.clear();
}
