#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildVoronoiDefinition();
std::unique_ptr<NodeKernel> createVoronoiKernel();

#include "NodeGraphKernels.hpp"

class NodeVoronoi final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* Voronoi = "voronoi";
}
