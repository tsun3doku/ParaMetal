#pragma once

#include "NodeGraphKernels.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

class NodeTransform final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    bool computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const override;

    static glm::mat4 buildLocalTransform(const NodeGraphNode& node);
    static std::array<float, 16> buildLocalTransformArray(const NodeGraphNode& node);

private:
    static void combineTransformParams(const NodeGraphNode& node, uint64_t& outHash);
};
