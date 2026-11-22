#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildCFDDefinition();
std::unique_ptr<NodeKernel> createCFDKernel();

#include "NodeGraphKernels.hpp"

class NodeCFD final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* CFD = "cfd";
}
