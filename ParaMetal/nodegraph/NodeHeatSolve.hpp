#pragma once

#include "NodeGraphKernels.hpp"

class NodeHeatSolve final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    HashValues computeOutputHashes(const NodeGraphKernelHashContext& context) const override;

    static NodeGraphNodeId selectHeatSolveNode(const NodeGraphState& state);
};
