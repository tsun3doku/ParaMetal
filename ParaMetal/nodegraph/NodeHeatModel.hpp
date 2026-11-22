#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildHeatModelDefinition();
std::unique_ptr<NodeKernel> createHeatModelKernel();

#include "NodeGraphKernels.hpp"

class NodeHeatModel final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* HeatModel = "heat_model";
}
