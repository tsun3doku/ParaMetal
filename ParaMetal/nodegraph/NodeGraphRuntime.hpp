#pragma once

#include "NodeGraphDataTypes.hpp"
#include "NodeGraphEvaluatedTypes.hpp"
#include "NodeGraphNodeState.hpp"
#include "NodeGraphState.hpp"
#include "NodeGraphKernels.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class NodeGraphRuntime {
public:
    explicit NodeGraphRuntime(const NodeRuntimeServices& services = {});
    ~NodeGraphRuntime();

    void applyDelta(const NodeGraphDelta& delta);
    void tick(const NodeGraphCompiled& compiled);
    void setOutputProductHandle(uint64_t socketKey, const ProductHandle& productHandle);

    const NodeGraphState& state() const {
        return graphState;
    }

    const NodeGraphEvaluationState& evaluationState() const {
        return currentEvaluationState;
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

    void applyChange(const NodeGraphChange& change);
    void execute(const NodeGraphCompiled& compiled);

    EvaluatedNodeInputs evaluateNodeInputs(
        const NodeGraphNode& node,
        const NodeGraphState& graphState,
        const NodeGraphEvaluationState& state) const;

    void publishOutputs(
        const NodeGraphNode& node,
        const std::vector<NodeDataBlock>& outputs,
        NodeGraphEvaluationState& state,
        bool frozen) const;
    bool publishCachedOutputs(
        const NodeGraphNode& node,
        NodeGraphEvaluationState& state) const;
    void publishBlockedOutputs(
        const NodeGraphNode& node,
        EvaluatedSocketStatus status,
        const std::string& error,
        NodeGraphEvaluationState& state) const;

    void evaluateLiveNode(
        const NodeGraphNode& node,
        const EvaluatedNodeInputs& inputs,
        NodeGraphEvaluationState& state);

    NodeRuntimeServices runtimeServices{};
    NodeGraphKernels kernels;
    NodeGraphState graphState{};
    NodeGraphEvaluationState currentEvaluationState{};
    std::unordered_map<uint32_t, CachedNodeOutputs> cachedOutputsByNodeId{};
    std::unordered_map<uint64_t, ProductHandle> productBySocket{};
};
