#pragma once

#include "NodeGraphTypes.hpp"
#include "runtime/RuntimeECS.hpp"

#include <unordered_set>
#include <vector>

class NodePayloadRegistry;
struct NodeDataBlock;
struct NodeGraphEvaluationState;

class NodeGraphDisplay {
public:
    std::unordered_set<uint64_t> computeDisplayKeys(
        const NodeGraphState& graphState,
        const NodeGraphEvaluationState& evaluationState,
        const ECSRegistry& registry,
        const NodePayloadRegistry* payloadRegistry) const;
private:
    void addDisplayKeys(
        uint64_t socketKey,
        const NodeDataBlock* block,
        const ECSRegistry& registry,
        const NodePayloadRegistry* payloadRegistry,
        std::unordered_set<uint64_t>& selectedKeys) const;
};
