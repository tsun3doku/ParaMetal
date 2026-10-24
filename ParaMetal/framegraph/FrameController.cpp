#include "FrameController.hpp"

#include "scene/CameraController.hpp"
#include "scene/NavigationGizmoController.hpp"
#include "FrameStats.hpp"
#include "FrameSync.hpp"
#include "render/SceneRenderer.hpp"
#include "app/ViewportTarget.hpp"
#include "VkFrameGraphBackend.hpp"
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
    VkFrameGraphBackend& frameGraphBackend,
    SceneRenderer& sceneRenderer,
    FrameSync& frameSync,
    FrameStats& frameStats,
    CameraController& cameraController,
    FrameControllerServices services,
    std::atomic<bool>& isOperating,
    std::atomic<bool>& isShuttingDown)
    : windowState(windowState),
      viewportTarget(viewportTarget),
      sceneRenderer(sceneRenderer),
      frameSync(frameSync),
      frameStats(frameStats),
      cameraController(cameraController),
      navigationGizmoController(services.navigationGizmoController),
      isOperating(isOperating),
      isShuttingDown(isShuttingDown),
      vulkanDevice(vulkanDevice),
      frameGraphRuntime(frameGraphBackend.getRuntime()),
      frameUpdateStage(
          services.inputController,
          services.uniformBufferManager,
          services.modelRegistry,
          services.lightingSystem,
          services.materialSystem,
          sceneRenderer,
          services.modelSelection),
      frameGraphicsStage(
          sceneRenderer,
          services.modelSelection,
          services.gizmoController,
          services.navigationGizmoController,
          services.wireframeRenderer) {
}

bool FrameController::drawFrame(VkCommandBuffer commandBuffer, uint32_t frameIndex, const render::RenderFlags& flags) {
    if (isShuttingDown.load(std::memory_order_acquire) || isOperating.load(std::memory_order_acquire)) {
        return false;
    }

    const VkExtent2D extent = viewportTarget.getExtent();
    if (commandBuffer == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0) {
        return false;
    }

    FrameState frameState{};
    frameState.frameIndex = frameIndex;
    std::vector<std::string> timingLines;
    frameUpdateStage.processPicking(frameState.frameIndex);

    frameState.extent = extent;
    navigationGizmoController.setViewport(frameState.extent, windowState.devicePixelRatio.load(std::memory_order_acquire));
    frameState.sceneView = cameraController.buildSceneView(frameState.extent);
    frameState.flags = flags;
    frameUpdateStage.updateFrameState(frameState.frameIndex, frameState.sceneView);

    if (flags.drawTimingOverlay) {
        timingLines = buildFrameTimingLines(frameState.frameIndex, frameStats.getComputeGpuMs());
    }

    const bool computePresent = frameSync.wasComputePresentedThisLap();

    if (computePresent && vulkanDevice.getComputeQueue() != vulkanDevice.getGraphicsQueue()) {
        std::cerr << "[FrameController] compute and graphics queues differ; no cross-queue dependency exists" << std::endl;
        isShuttingDown.store(true, std::memory_order_release);
        return false;
    }

    FrameSyncState syncState{};
    syncState.insertComputeToGraphicsBarrier = computePresent;
    syncState.waitDstStageMask = frameGraphRuntime.getComputeToGraphicsWaitDstStageMask();

    updateTimingOverlay(timingLines, frameState.flags);

    if (!frameGraphicsStage.collectExternal(commandBuffer, frameState, syncState)) {
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
