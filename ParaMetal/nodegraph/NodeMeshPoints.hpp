#pragma once

#include "NodeGraphKernels.hpp"

class NodeMeshPoints : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    bool computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const override;
};
