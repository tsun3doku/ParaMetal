#pragma once

#include "NodeGraphKernels.hpp"

class NodePoints : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    HashValues computeOutputHashes(const NodeGraphKernelHashContext& context) const override;
};
