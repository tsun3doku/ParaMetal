#include "FrameController.hpp"

#include "scene/CameraController.hpp"
#include "interaction/InteractionContext.hpp"
#include "runtime/SceneContext.hpp"
#include "FrameStats.hpp"
#include "FrameSync.hpp"
#include "render/SceneRenderer.hpp"
#include "app/ViewportTarget.hpp"
#include "VkFrameGraphRuntime.hpp"
#include "render/RenderConfig.hpp"
#include "render/WindowRuntimeState.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

FrameController::FrameController(
    const WindowRuntimeState& windowState,
    VulkanDevice& vulkanDevice,
    ViewportTarget& viewportTarget,
    VkFrameGraphRuntime& frameGraphRuntime,
    SceneRenderer& sceneRenderer,
    FrameSync& frameSync,
    FrameStats& frameStats,
    SceneContext& scene,
    InteractionContext& interaction,
    ModelSelection& modelSelection,
    WireframeRenderer& wireframeRenderer,
    std::atomic<bool>& isOperating,
    std::atomic<bool>& isShuttingDown)
    : windowState(windowState),
      viewportTarget(viewportTarget),
      sceneRenderer(sceneRenderer),
      modelSelection(modelSelection),
      wireframeRenderer(wireframeRenderer),
      frameSync(frameSync),
      frameStats(frameStats),
      cameraController(scene.cameraController()),
      interaction(interaction),
      isOperating(isOperating),
      isShuttingDown(isShuttingDown),
      frameUpdateStage(
          interaction,
          *scene.uniformBufferManager(),
          *scene.modelRegistry(),
          *scene.lightingSystem(),
          *scene.materialSystem(),
          sceneRenderer,
          modelSelection),
      vulkanDevice(vulkanDevice),
      frameGraphRuntime(frameGraphRuntime) {
}

bool FrameController::drawFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex, const render::RenderFlags& flags) {
    if (isShuttingDown.load(std::memory_order_acquire) || isOperating.load(std::memory_order_acquire)) {
        return false;
    }

    const VkExtent2D extent = viewportTarget.getExtent();
    if (commandBuffer == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0) {
        return false;
    }

    std::vector<std::string> timingLines;
    frameUpdateStage.processPicking(frameIndex);

    interaction.setViewport(extent, windowState.devicePixelRatio.load(std::memory_order_acquire));
    const render::SceneView sceneView = cameraController.buildSceneView(extent);
    frameUpdateStage.updateFrameState(frameIndex, sceneView);

    if (flags.drawTimingOverlay) {
        timingLines = buildFrameTimingLines(frameIndex, frameStats.getComputeGpuMs());
    }

    const bool computePresent = frameSync.wasComputePresentedThisLap();

    if (computePresent && vulkanDevice.getComputeQueue() != vulkanDevice.getGraphicsQueue()) {
        std::cerr << "[FrameController] compute and graphics queues differ; no cross-queue dependency exists" << std::endl;
        isShuttingDown.store(true, std::memory_order_release);
        return false;
    }

    updateTimingOverlay(timingLines, flags);

    render::RenderFrameRequest frameRequest{};
    frameRequest.frameIndex = frameIndex;
    frameRequest.imageIndex = 0;
    frameRequest.extent = extent;
    frameRequest.sceneView = sceneView;
    frameRequest.flags = flags;

    render::RenderServices services{};
    services.modelSelection = &modelSelection;
    services.gizmoController = &interaction.gizmoController();
    services.navigationGizmoController = &interaction.navigationGizmoController();
    services.wireframeRenderer = &wireframeRenderer;

    if (!sceneRenderer.recordExternalCommandBuffer(
            commandBuffer,
            frameRequest,
            services,
            computePresent,
            frameGraphRuntime.getComputeToGraphicsWaitDstStageMask())) {
        isShuttingDown.store(true, std::memory_order_release);
        return false;
    }

    return true;
}

std::vector<std::string> FrameController::buildFrameTimingLines(uint32_t frameIndex, std::optional<float> computeGpuMs) {
    GpuTimingStats gpuTiming{};
    const GpuTimingStats* graphicsTiming = nullptr;
    if (sceneRenderer.getGpuTimingStats(frameIndex, gpuTiming)) {
        graphicsTiming = &gpuTiming;
    }

    std::vector<std::string> lines = frameStats.buildTimingLines(graphicsTiming, computeGpuMs);

    std::ostringstream fpsLine;
    fpsLine << std::fixed << std::setprecision(1)
            << "FPS: " << frameStats.getSwapFpsValue();
    lines.push_back(fpsLine.str());

    return lines;
}

void FrameController::updateTimingOverlay(std::vector<std::string>& timingLines, const render::RenderFlags& flags) {
    if (flags.drawTimingOverlay) {
        frameStats.capitalizeLines(timingLines);
        sceneRenderer.setTimingOverlayLines(timingLines);
        return;
    }

    sceneRenderer.setTimingOverlayLines({});
}
