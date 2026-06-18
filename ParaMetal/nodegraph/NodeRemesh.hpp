#pragma once

#include "NodeGraphKernels.hpp"

class NodeRemesh final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    HashValues computeOutputHashes(const NodeGraphKernelHashContext& context) const override;
};
