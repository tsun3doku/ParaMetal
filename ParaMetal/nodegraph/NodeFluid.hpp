#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildFluidDefinition();
std::unique_ptr<NodeKernel> createFluidKernel();

#include "NodeGraphKernels.hpp"

class NodeFluid final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

using NodeHeatFluid = NodeFluid;

namespace nodegraphtypes {
inline constexpr const char* Fluid = "fluid";
}
