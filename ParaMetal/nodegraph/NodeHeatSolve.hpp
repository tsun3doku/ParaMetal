#pragma once

#include "NodeGraphKernels.hpp"

class NodeHeatSolve final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;

    static NodeGraphNodeId selectHeatSolveNode(const NodeGraphState& state);
};
