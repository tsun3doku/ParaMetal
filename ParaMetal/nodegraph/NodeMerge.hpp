#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildMergeDefinition();
std::unique_ptr<NodeKernel> createMergeKernel();

#include "NodeGraphKernels.hpp"

class NodeMerge : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* Merge = "merge";
}
