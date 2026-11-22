#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildBoundaryConditionsDefinition();
std::unique_ptr<NodeKernel> createBoundaryConditionsKernel();

#include "NodeGraphKernels.hpp"

class NodeBoundaryConditions final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* BoundaryConditions = "boundary_conditions";
}
