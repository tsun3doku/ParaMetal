#pragma once

#include "FrameTypes.hpp"

class SceneRenderer;
class ModelSelection;
class GizmoController;
class NavigationGizmoController;
class WireframeRenderer;

// Records the scene external command buffer without submitting
class FrameGraphicsStage {
public:
    FrameGraphicsStage(
        SceneRenderer& sceneRenderer,
        ModelSelection& modelSelection,
        GizmoController& gizmoController,
        NavigationGizmoController& navigationGizmoController,
        WireframeRenderer& wireframeRenderer);

    bool collectExternal(
        VkCommandBuffer commandBuffer,
        const FrameState& frameState,
        const FrameSyncState& syncState);

private:
    SceneRenderer& sceneRenderer;
    ModelSelection& modelSelection;
    GizmoController& gizmoController;
    NavigationGizmoController& navigationGizmoController;
    WireframeRenderer& wireframeRenderer;
};
