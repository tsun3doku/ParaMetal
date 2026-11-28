#include "NodeGraphRuntime.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodePayloadRegistry.hpp"

#include <algorithm>
#include <utility>
#include <vector>

NodeGraphRuntime::NodeGraphRuntime(NodePayloadRegistry* registry)
    : payloadRegistry(registry) {
}

NodeGraphRuntime::~NodeGraphRuntime() = default;

void NodeGraphRuntime::publishOutputs(
    const NodeGraphNode& node,
    const std::vector<NodeDataBlock>& outputs,
    NodeGraphEvaluation& state,
    bool frozen) const {
    const std::size_t count = std::min(outputs.size(), node.outputs.size());
    for (std::size_t i = 0; i < count; ++i) {
        EvaluatedSocketValue value{};
        value.status = EvaluatedSocketStatus::Value;
        value.data = outputs[i];
        value.data.isFrozen = frozen;
        state.outputsBySocket[NodeSocketKey(node.id, node.outputs[i].id).value] = value;
    }
}

bool NodeGraphRuntime::publishCachedOutputs(
    const NodeGraphNode& node,
    NodeGraphEvaluation& state) const {
    const auto it = cachedOutputsByNodeId.find(node.id.value);
    if (it == cachedOutputsByNodeId.end() ||
        it->second.outputs.size() != node.outputs.size()) {
        return false;
    }

    publishOutputs(node, it->second.outputs, state, it->second.pinned);
    return true;
}

void NodeGraphRuntime::publishBlockedOutputs(
    const NodeGraphNode& node,
    EvaluatedSocketStatus status,
    const std::string& error,
    NodeGraphEvaluation& state,
    bool backpressured) const {
    EvaluatedSocketValue blockedValue{};
    blockedValue.status = status;
    blockedValue.backpressured = backpressured;
    if (status == EvaluatedSocketStatus::Error) {
        blockedValue.error = error;
    }
    for (const NodeGraphSocket& outputSocket : node.outputs) {
        state.outputsBySocket[NodeSocketKey(node.id, outputSocket.id).value] = blockedValue;
    }
}

NodeGraphRuntime::EvaluatedNodeInputs NodeGraphRuntime::evaluateNodeInputs(
    const NodeGraphNode& node,
    const NodeGraphState& graphState,
    const NodeGraphEvaluation& state) const {

    EvaluatedNodeInputs result;
    result.values.assign(node.inputs.size(), {});

    for (std::size_t inputIndex = 0; inputIndex < node.inputs.size(); ++inputIndex) {
        const NodeGraphSocket& inputSocket = node.inputs[inputIndex];
        const std::vector<const NodeGraphEdge*> incomingEdges =
            graphState.edges.incoming(node.id, inputSocket.id);

        if (incomingEdges.empty()) {
            if (!inputSocket.required) {
                continue;
            }
            result.status = EvaluatedSocketStatus::Missing;
            return result;
        }

        for (const NodeGraphEdge* edgePtr : incomingEdges) {
            if (!edgePtr) {
                continue;
            }
            const NodeGraphEdge& edge = *edgePtr;
            const auto outputIt = state.outputsBySocket.find(NodeSocketKey(edge.fromNode, edge.fromSocket).value);
            if (outputIt == state.outputsBySocket.end()) {
                if (!inputSocket.required) {
                    continue;
                }
                result.status = EvaluatedSocketStatus::Missing;
                return result;
            }

            const EvaluatedSocketValue& inputValue = outputIt->second;
            if (inputValue.status != EvaluatedSocketStatus::Value) {
                result.status = inputValue.status;
                result.error = inputValue.error;
                return result;
            }

            result.values[inputIndex].push_back(&inputValue.data);
        }
    }

    return result;
}

void NodeGraphRuntime::applyDelta(const NodeGraphDelta& delta) {
    pendingCacheHashes.clear();
    if (!delta.changes.empty()) {
        bool topologyChanged = false;
        for (const NodeGraphChange& change : delta.changes) {
            if (change.reason == NodeGraphChangeReason::Topology) {
                topologyChanged = true;
            }
            applyNodeGraphChange(graphState, change);
            if (change.type == NodeGraphChangeType::Reset) {
                cachedOutputsByNodeId.clear();
            } else if (change.type == NodeGraphChangeType::NodeRemoved) {
                cachedOutputsByNodeId.erase(change.nodeId.value);
            }
        }
        if (topologyChanged) {
            for (auto it = cachedOutputsByNodeId.begin(); it != cachedOutputsByNodeId.end();) {
                if (it->second.pinned) {
                    ++it;
                } else {
                    it = cachedOutputsByNodeId.erase(it);
                }
            }
            currentEvaluation.outputsBySocket.clear();
        }
    }
    graphState.instanceId = delta.graphInstanceId;
    graphState.revision = delta.toRevision;
    syncPayloadRoots();
}

void NodeGraphRuntime::beginEvaluation(const NodeGraphCompiled& compiled) {
    pendingCacheHashes.clear();
    if (graphState.nodes.size() > 0 && compiled.executionOrder.size() != graphState.nodes.size()) {
        currentEvaluation.outputsBySocket.clear();
        return;
    }
    currentEvaluation.outputsBySocket.clear();
    currentEvaluation.outputsBySocket.reserve(graphState.edges.size() * 2);
    syncPayloadRoots();
}

void NodeGraphRuntime::syncPayloadRoots() {
    if (!payloadRegistry) {
        return;
    }
    std::vector<NodeDataHandle> roots;
    roots.reserve(currentEvaluation.outputsBySocket.size() + cachedOutputsByNodeId.size());
    for (const auto& [socketKey, value] : currentEvaluation.outputsBySocket) {
        (void)socketKey;
        if (value.status == EvaluatedSocketStatus::Value &&
            value.data.payloadHandle.key != 0) {
            roots.push_back(value.data.payloadHandle);
        }
    }
    for (const auto& [nodeId, cached] : cachedOutputsByNodeId) {
        (void)nodeId;
        for (const NodeDataBlock& output : cached.outputs) {
            if (output.payloadHandle.key != 0) {
                roots.push_back(output.payloadHandle);
            }
        }
    }
    payloadRegistry->setLiveRoots(std::move(roots));
    payloadRegistry->reclaimUnreachable();
}

const NodeDataBlock* NodeGraphRuntime::cachedOutput(
    NodeGraphNodeId nodeId, NodeGraphSocketId socketId) const {
    const NodeGraphNode* node = graphState.node(nodeId);
    const auto cached = cachedOutputsByNodeId.find(nodeId.value);
    if (!node || cached == cachedOutputsByNodeId.end()) return nullptr;
    for (std::size_t i = 0; i < node->outputs.size(); ++i) {
        if (node->outputs[i].id == socketId && i < cached->second.outputs.size()) {
            return &cached->second.outputs[i];
        }
    }
    return nullptr;
}

bool NodeGraphRuntime::commitNode(NodeGraphNodeId nodeId) {
    const auto pending = pendingCacheHashes.find(nodeId.value);
    if (pending == pendingCacheHashes.end()) return true; // Cache hit or kernelless node.
    const NodeGraphNode* node = graphState.node(nodeId);
    if (!node) return false;

    CachedNodeOutputs completed{};
    completed.hash = pending->second;
    completed.outputs.reserve(node->outputs.size());
    for (const NodeGraphSocket& socket : node->outputs) {
        const EvaluatedSocketValue* value = currentEvaluation.outputFor(
            NodeSocketKey(nodeId, socket.id).value);
        if (!value || value->status != EvaluatedSocketStatus::Value) return false;
        completed.outputs.push_back(value->data);
    }
    cachedOutputsByNodeId.insert_or_assign(nodeId.value, std::move(completed));
    pendingCacheHashes.erase(pending);
    syncPayloadRoots();
    return true;
}

bool NodeGraphRuntime::replaceEvaluatedOutput(
    NodeGraphNodeId nodeId,
    NodeGraphSocketId socketId,
    const NodeDataBlock& output) {
    const NodeGraphNode* node = graphState.node(nodeId);
    if (!node) return false;
    const uint64_t socketKey = NodeSocketKey(nodeId, socketId).value;
    const auto evaluated = currentEvaluation.outputsBySocket.find(socketKey);
    if (evaluated == currentEvaluation.outputsBySocket.end() ||
        evaluated->second.status != EvaluatedSocketStatus::Value) return false;
    const NodeDataBlock& before = evaluated->second.data;
    if (before.payloadHandle == output.payloadHandle &&
        before.hashes.full == output.hashes.full &&
        before.hashes.geometry == output.hashes.geometry &&
        before.hashes.display == output.hashes.display) {
        return true;
    }
    if (pendingCacheHashes.find(nodeId.value) == pendingCacheHashes.end()) {
        const auto cached = cachedOutputsByNodeId.find(nodeId.value);
        if (cached != cachedOutputsByNodeId.end()) {
            pendingCacheHashes.emplace(nodeId.value, cached->second.hash);
        }
    }
    evaluated->second.data = output;
    syncPayloadRoots();
    return true;
}

void NodeGraphRuntime::failNode(NodeGraphNodeId nodeId, const std::string& error) {
    pendingCacheHashes.erase(nodeId.value);
    const NodeGraphNode* node = graphState.node(nodeId);
    if (!node) {
        return;
    }
    const auto cacheIt = cachedOutputsByNodeId.find(nodeId.value);
    if (cacheIt != cachedOutputsByNodeId.end() && !cacheIt->second.pinned) {
        cachedOutputsByNodeId.erase(cacheIt);
    }
    publishBlockedOutputs(*node, EvaluatedSocketStatus::Error, error, currentEvaluation);
    syncPayloadRoots();
}

bool NodeGraphRuntime::nodeBackpressured(NodeGraphNodeId nodeId) const {
    const NodeGraphNode* node = graphState.node(nodeId);
    if (!node) {
        return false;
    }
    for (const NodeGraphSocket& output : node->outputs) {
        const auto it = currentEvaluation.outputsBySocket.find(
            NodeSocketKey(node->id, output.id).value);
        if (it != currentEvaluation.outputsBySocket.end() && it->second.backpressured) {
            return true;
        }
    }
    return false;
}

void NodeGraphRuntime::evaluateNode(NodeGraphNodeId nodeId) {
    auto it = graphState.nodes.find(nodeId.value);
    if (it == graphState.nodes.end()) {
        return;
    }

    const NodeGraphNode& node = it->second;
    NodeGraphEvaluation& state = currentEvaluation;

    bool pinned = (node.state.frozenState() == NodeGraphNodeState::FrozenState::Frozen);
    if (!pinned) {
        for (NodeGraphNodeId upstreamNodeId : graphState.edges.upstreamNodes(node.id)) {
            auto cacheIt = cachedOutputsByNodeId.find(upstreamNodeId.value);
            if (cacheIt != cachedOutputsByNodeId.end() && cacheIt->second.pinned) {
                pinned = true;
                break;
            }
        }
    }

    // Pinned nodes bypass input evaluation and republish their cached output
    if (pinned) {
        auto cacheIt = cachedOutputsByNodeId.find(node.id.value);
        if (cacheIt != cachedOutputsByNodeId.end()) {
            cacheIt->second.pinned = true;
        }
        if (!publishCachedOutputs(node, state)) {
            publishBlockedOutputs(
                node,
                EvaluatedSocketStatus::Error,
                "Frozen node has no cached output.",
                state);
        }
        return;
    }

    const NodeTypeId typeId = getNodeTypeId(node.typeId);
    if (!kernels.hasKernel(typeId)) {
        if (kernels.requiresKernel(typeId)) {
            publishBlockedOutputs(
                node,
                EvaluatedSocketStatus::Error,
                "No kernel registered for node type '" + typeId + "'.",
                state);
        }
        return;
    }

    // Evaluate inputs
    EvaluatedNodeInputs inputs = evaluateNodeInputs(node, graphState, state);
    if (!inputs.ready()) {
        publishBlockedOutputs(node, inputs.status, inputs.error, state);
        return;
    }

    // Live evaluation
    evaluateLiveNode(node, inputs, state);
}

void NodeGraphRuntime::evaluateLiveNode(
    const NodeGraphNode& node,
    const EvaluatedNodeInputs& inputs,
    NodeGraphEvaluation& state) {

    const NodeKernelRuntime kernelRuntime{
        graphState,
        payloadRegistry};

    HashValues outputHashes = kernels.computeOutputHashes(node, kernelRuntime, inputs.values);
    uint64_t cacheHash = outputHashes.full;

    // Cache hit
    const auto cacheIt = cachedOutputsByNodeId.find(node.id.value);
    if (cacheIt != cachedOutputsByNodeId.end() &&
        cacheIt->second.hash == cacheHash &&
        cacheIt->second.outputs.size() == node.outputs.size()) {
        cacheIt->second.pinned = false;
        publishOutputs(node, cacheIt->second.outputs, state, /*frozen=*/false);
        return;
    }

    std::vector<NodeDataBlock> outputValues(node.outputs.size());

    std::string kernelError;
    kernels.executeNode(node, kernelRuntime, inputs.values, outputValues, outputHashes, kernelError);
    if (!kernelError.empty()) {
        if (kernelError != "Payload publication backpressured.") {
            cachedOutputsByNodeId.erase(node.id.value);
        }
        publishBlockedOutputs(
            node,
            EvaluatedSocketStatus::Error,
            kernelError,
            state,
            kernelError == "Payload publication backpressured.");
        return;
    }

    for (NodeDataBlock& output : outputValues) {
        output.hashes = outputHashes;
    }

    // Provisional output is visible to compilation, but not committed to the cache.
    // Runtime publication must succeed before the old completed cache is replaced.
    pendingCacheHashes.insert_or_assign(node.id.value, cacheHash);
    publishOutputs(node, outputValues, state, /*frozen=*/false);
    syncPayloadRoots();
}
