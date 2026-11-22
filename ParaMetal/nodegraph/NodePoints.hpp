#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildPointsDefinition();
std::unique_ptr<NodeKernel> createPointsKernel();

#include "NodeGraphKernels.hpp"

class NodePoints : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* Points = "points";
}
