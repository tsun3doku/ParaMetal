#pragma once

#include <cstdint>

#include "FrameTypes.hpp"

class VulkanDevice;
class FrameSync;
class SceneRenderer;
class ModelSelection;
class GizmoController;
class NavigationGizmoController;
class WireframeRenderer;

// Collected graphics work for a frame. VK_NULL_HANDLE commandBuffer means
// recording failed; FrameController treats that as a skip/recreate.
struct FrameGraphicsCollection {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    FrameStageResult result = FrameStageResult::Continue;
};

class FrameGraphicsStage {
public:
    FrameGraphicsStage(
        VulkanDevice& vulkanDevice,
        FrameSync& frameSync,
        SceneRenderer& sceneRenderer,
        ModelSelection& modelSelection,
        GizmoController& gizmoController,
        NavigationGizmoController& navigationGizmoController,
        WireframeRenderer& wireframeRenderer);

    // Records the scene command buffer. Does NOT submit. Returns the collected
    // work; result indicates whether to proceed, recreate, etc.
    FrameGraphicsCollection collect(const FrameState& frameState, const FrameSyncState& syncState);

private:
    VulkanDevice& vulkanDevice;
    FrameSync& frameSync;
    SceneRenderer& sceneRenderer;
    ModelSelection& modelSelection;
    GizmoController& gizmoController;
    NavigationGizmoController& navigationGizmoController;
    WireframeRenderer& wireframeRenderer;
};
