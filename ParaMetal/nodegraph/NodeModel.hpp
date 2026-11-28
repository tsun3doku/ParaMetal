#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildModelDefinition();
std::unique_ptr<NodeKernel> createModelKernel();

class NodeModel final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* Model = "model";
}
