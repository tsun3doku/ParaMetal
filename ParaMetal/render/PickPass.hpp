#pragma once

#include "framegraph/FramePass.hpp"
#include "framegraph/FrameGraphTypes.hpp"

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class IntrinsicRenderer;
class ModelRegistry;
class UniformBufferManager;
class VulkanDevice;
class VkFrameGraphRuntime;
class GizmoRenderer;

namespace render {

class GeometryPass;

class PickPass : public Pass {
public:
    PickPass(
        VulkanDevice& device,
        VkFrameGraphRuntime& frameGraphRuntime,
        ModelRegistry& resources,
        GeometryPass& geometry,
        GizmoRenderer& gizmoRenderer,
        IntrinsicRenderer& intrinsicRenderer,
        framegraph::PassId passId);

    const char* name() const override;
    bool create() override;
    void resize(VkExtent2D extent) override;
    void updateDescriptors() override;
    void record(const FrameContext& context, const SceneView& view, const RenderFlags& flags, RenderServices& services) override;
    void destroy() override;

private:
    struct PickPushConstant {
        alignas(16) glm::mat4 modelMatrix;
        uint32_t pickId = 0;
        uint32_t elementMode = 0;
        uint32_t primitiveBase = 0;
        uint32_t padding = 0;
    };

    bool createModelPipeline();
    void drawOccluders(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const std::vector<uint32_t>& modelIds);

    VulkanDevice& vulkanDevice;
    VkFrameGraphRuntime& frameGraphRuntime;
    ModelRegistry& modelRegistry;
    GeometryPass& geometryPass;
    GizmoRenderer& gizmoRenderer;
    IntrinsicRenderer& intrinsicRenderer;
    framegraph::PassId passId{};

    VkPipelineLayout modelPipelineLayout = VK_NULL_HANDLE;
    VkPipeline modelPipeline = VK_NULL_HANDLE;
    VkPipeline occluderPipeline = VK_NULL_HANDLE;
    VkPipeline subselectionPipeline = VK_NULL_HANDLE;
    bool ready = false;
};

}
