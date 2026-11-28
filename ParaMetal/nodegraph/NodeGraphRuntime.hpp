#pragma once

#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphEvaluation.hpp"
#include "NodeGraphNodeState.hpp"
#include "NodeGraphState.hpp"
#include "NodeGraphKernels.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class NodeGraphRuntime {
public:
    explicit NodeGraphRuntime(NodePayloadRegistry* payloadRegistry = nullptr);
    ~NodeGraphRuntime();

    void applyDelta(const NodeGraphDelta& delta);
    void beginEvaluation(const NodeGraphCompiled& compiled);
    void evaluateNode(NodeGraphNodeId nodeId);
    // Cache only after the node's runtime operation and output publication succeed.
    bool commitNode(NodeGraphNodeId nodeId);
    const NodeDataBlock* cachedOutput(NodeGraphNodeId nodeId, NodeGraphSocketId socketId) const;
    bool replaceEvaluatedOutput(
        NodeGraphNodeId nodeId,
        NodeGraphSocketId socketId,
        const NodeDataBlock& output);
    void failNode(NodeGraphNodeId nodeId, const std::string& error);
    bool nodeBackpressured(NodeGraphNodeId nodeId) const;
    void syncPayloadRoots();
    bool isValid() const { return kernels.isValid(); }

    const NodeGraphState& state() const {
        return graphState;
    }

    const NodeGraphEvaluation& evaluation() const {
        return currentEvaluation;
    }

    const NodeKernel* kernelFor(const NodeTypeId& typeId) const {
        return kernels.findKernel(typeId);
    }

private:
    struct CachedNodeOutputs {
        uint64_t hash = 0;
        std::vector<NodeDataBlock> outputs;
        bool pinned = false;
    };

    struct EvaluatedNodeInputs {
        std::vector<std::vector<const NodeDataBlock*>> values;
        EvaluatedSocketStatus status = EvaluatedSocketStatus::Value;
        std::string error;

        bool ready() const {
            return status == EvaluatedSocketStatus::Value;
        }
    };

    EvaluatedNodeInputs evaluateNodeInputs(
        const NodeGraphNode& node,
        const NodeGraphState& graphState,
        const NodeGraphEvaluation& state) const;

    void publishOutputs(
        const NodeGraphNode& node,
        const std::vector<NodeDataBlock>& outputs,
        NodeGraphEvaluation& state,
        bool frozen) const;
    bool publishCachedOutputs(
        const NodeGraphNode& node,
        NodeGraphEvaluation& state) const;
    void publishBlockedOutputs(
        const NodeGraphNode& node,
        EvaluatedSocketStatus status,
        const std::string& error,
        NodeGraphEvaluation& state,
        bool backpressured = false) const;

    void evaluateLiveNode(
        const NodeGraphNode& node,
        const EvaluatedNodeInputs& inputs,
        NodeGraphEvaluation& state);

    NodePayloadRegistry* payloadRegistry = nullptr;
    NodeGraphKernels kernels;
    NodeGraphState graphState{};
    NodeGraphEvaluation currentEvaluation{};
    std::unordered_map<uint32_t, CachedNodeOutputs> cachedOutputsByNodeId{};
    std::unordered_map<uint32_t, uint64_t> pendingCacheHashes;
};
