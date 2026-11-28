#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildTransformDefinition();
std::unique_ptr<NodeKernel> createTransformKernel();

#include "NodeTransformParams.hpp"
#include "util/Units.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>

struct NodeTransformEditState {
    TransformNodeParams start{};
    TransformNodeParams preview{};
    glm::mat4 parentWorld{1.0f};
    float canonicalToWorldScale = 1.0f;
    int mode = 0;
    int axis = 0;
};

class NodeTransform final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;

    static glm::mat4 buildLocalTransform(const TransformNodeParams& params);
    static glm::mat4 buildLocalTransform(const NodeGraphNode& node);
    static std::array<float, 16> buildLocalTransformArray(const NodeGraphNode& node);
    static bool beginEdit(
        const NodeGraphNode& node,
        const glm::mat4& currentWorld,
        units::LengthUnit worldUnit,
        int mode,
        int axis,
        NodeTransformEditState& outState);
    static bool previewEdit(
        NodeTransformEditState& state,
        const glm::vec3& translationDeltaWorld,
        float rotationDeltaDegrees,
        glm::mat4& outWorld);
    static bool buildEditParameters(
        const NodeTransformEditState& state,
        std::vector<NodeGraphParamValue>& outParameters);

private:
    static void combineTransformParams(const NodeGraphNode& node, uint64_t& outHash);
};

namespace nodegraphtypes {
inline constexpr const char* Transform = "transform";
}
