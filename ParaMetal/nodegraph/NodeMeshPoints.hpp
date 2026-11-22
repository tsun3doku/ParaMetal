#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildMeshPointsDefinition();
std::unique_ptr<NodeKernel> createMeshPointsKernel();

#include "NodeGraphKernels.hpp"

class NodeMeshPoints : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* MeshPoints = "mesh_points";
}
