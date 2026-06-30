#include "FrameGraphicsStage.hpp"

#include <iostream>

#include "FramePass.hpp"
#include "FrameSync.hpp"
#include "render/SceneRenderer.hpp"
#include "vulkan/VulkanDevice.hpp"

FrameGraphicsStage::FrameGraphicsStage(
    VulkanDevice& vulkanDevice,
    FrameSync& frameSync,
    SceneRenderer& sceneRenderer,
    ModelSelection& modelSelection,
    GizmoController& gizmoController,
    WireframeRenderer& wireframeRenderer)
    : vulkanDevice(vulkanDevice),
      frameSync(frameSync),
      sceneRenderer(sceneRenderer),
      modelSelection(modelSelection),
      gizmoController(gizmoController),
      wireframeRenderer(wireframeRenderer) {
}

FrameGraphicsCollection FrameGraphicsStage::collect(const FrameState& frameState, const FrameSyncState& syncState) {
    FrameGraphicsCollection collection{};

    const auto& commandBuffers = sceneRenderer.getCommandBuffers();
    if (frameState.frameIndex >= commandBuffers.size()) {
        std::cerr << "[FrameGraphicsStage] Missing scene renderer command buffer for frame index" << std::endl;
        collection.result = FrameStageResult::Fatal;
        return collection;
    }

    VkCommandBuffer commandBuffer = commandBuffers[frameState.frameIndex];
    vkResetCommandBuffer(commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

    render::RenderFrameRequest frameRequest{};
    frameRequest.frameIndex = frameState.frameIndex;
    frameRequest.imageIndex = frameState.imageIndex;
    frameRequest.extent = frameState.extent;
    frameRequest.sceneView = frameState.sceneView;
    frameRequest.flags = frameState.flags;

    render::RenderServices services{};
    services.modelSelection = &modelSelection;
    services.gizmoController = &gizmoController;
    services.wireframeRenderer = &wireframeRenderer;

    if (!sceneRenderer.recordCommandBuffer(
        frameRequest,
        services,
        syncState.insertComputeToGraphicsBarrier,
        syncState.waitDstStageMask)) {
        std::cerr << "[FrameGraphicsStage] Scene command recording failed. Triggering swapchain recreation." << std::endl;
        collection.result = FrameStageResult::RecreateSwapchain;
        return collection;
    }

    collection.commandBuffer = commandBuffer;
    collection.result = FrameStageResult::Continue;
    return collection;
}
