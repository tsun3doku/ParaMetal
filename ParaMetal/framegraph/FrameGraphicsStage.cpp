#include "FrameGraphicsStage.hpp"

#include "FramePass.hpp"
#include "render/SceneRenderer.hpp"

FrameGraphicsStage::FrameGraphicsStage(
    SceneRenderer& sceneRenderer,
    ModelSelection& modelSelection,
    GizmoController& gizmoController,
    NavigationGizmoController& navigationGizmoController,
    WireframeRenderer& wireframeRenderer)
    : sceneRenderer(sceneRenderer),
      modelSelection(modelSelection),
      gizmoController(gizmoController),
      navigationGizmoController(navigationGizmoController),
      wireframeRenderer(wireframeRenderer) {
}

bool FrameGraphicsStage::collectExternal(
    VkCommandBuffer commandBuffer,
    const FrameState& frameState,
    const FrameSyncState& syncState) {
    if (commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    render::RenderFrameRequest frameRequest{};
    frameRequest.frameIndex = frameState.frameIndex;
    frameRequest.imageIndex = 0;
    frameRequest.extent = frameState.extent;
    frameRequest.sceneView = frameState.sceneView;
    frameRequest.flags = frameState.flags;

    render::RenderServices services{};
    services.modelSelection = &modelSelection;
    services.gizmoController = &gizmoController;
    services.navigationGizmoController = &navigationGizmoController;
    services.wireframeRenderer = &wireframeRenderer;

    return sceneRenderer.recordExternalCommandBuffer(
        commandBuffer,
        frameRequest,
        services,
        syncState.insertComputeToGraphicsBarrier,
        syncState.waitDstStageMask);
}
