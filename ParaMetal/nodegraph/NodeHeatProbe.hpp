#pragma once

#include "NodeGraphKernels.hpp"
#include "NodeHeatProbeParams.hpp"

NodeTypeDefinition buildHeatProbeDefinition();
std::unique_ptr<NodeKernel> createHeatProbeKernel();

class NodeHeatProbe final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};
