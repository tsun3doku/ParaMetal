#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildHeatSolveDefinition();
std::unique_ptr<NodeKernel> createHeatSolveKernel();

#include "NodeGraphKernels.hpp"

class NodeHeatSolve final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;

    static NodeGraphNodeId selectHeatSolveNode(const NodeGraphState& state);
};

namespace nodegraphtypes {
inline constexpr const char* HeatSolve = "heat_solve";
}
