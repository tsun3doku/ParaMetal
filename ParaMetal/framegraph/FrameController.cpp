#include "FrameController.hpp"

#include "scene/CameraController.hpp"
#include "FrameStats.hpp"
#include "FrameSync.hpp"
#include "render/SceneRenderer.hpp"
#include "app/SwapchainManager.hpp"
#include "VkFrameGraphBackend.hpp"
#include "framegraph/ComputePass.hpp"
#include "render/RenderConfig.hpp"
#include "render/WindowRuntimeState.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <optional>
#include <utility>
#include <iostream>

FrameController::FrameController(
    const WindowRuntimeState& windowState,
    VulkanDevice& vulkanDevice,
    SwapchainManager& swapchainManager,
    FrameGraph& frameGraph,
    VkFrameGraphBackend& frameGraphBackend,
    SceneRenderer& sceneRenderer,
    FrameSync& frameSync,
    FrameStats& frameStats,
    CameraController& cameraController,
    FrameControllerServices services,
    std::atomic<bool>& isOperating,
    std::atomic<bool>& isShuttingDown)
    : windowState(windowState),
      swapchainManager(swapchainManager),
      sceneRenderer(sceneRenderer),
      frameSync(frameSync),
      frameStats(frameStats),
      cameraController(cameraController),
      isOperating(isOperating),
      isShuttingDown(isShuttingDown),
      vulkanDevice(vulkanDevice),
      swapchainStage(
          windowState,
          vulkanDevice,
          swapchainManager,
          frameGraph,
          frameGraphBackend,
          sceneRenderer,
          frameSync,
          isShuttingDown),
      frameUpdateStage(
          services.inputController,
          services.uniformBufferManager,
          services.resourceManager,
          services.lightingSystem,
          services.materialSystem,
          sceneRenderer,
          services.modelSelection),
      frameComputeStage(
          vulkanDevice,
          frameGraphBackend.getRuntime(),
          frameSync),
      frameGraphicsStage(
          vulkanDevice,
          frameSync,
          sceneRenderer,
          services.modelSelection,
          services.gizmoController,
          services.wireframeRenderer) {
}

bool FrameController::initializeSyncObjects() {
    return swapchainStage.initializeSyncObjects();
}

void FrameController::shutdownSyncObjects() {
    swapchainStage.shutdownSyncObjects();
}

void FrameController::cleanupSwapChain() {
    swapchainStage.cleanupSwapChain();
}

bool FrameController::recreateSwapChain() {
    return swapchainStage.recreateSwapChain();
}

void FrameController::drawFrame(const render::RenderFlags& flags, const std::vector<ComputePass*>& computePasses) {
    static std::atomic<uint64_t> frameCallCounter{0};
    const uint64_t callIndex = frameCallCounter.fetch_add(1, std::memory_order_relaxed);

    if (isShuttingDown.load(std::memory_order_acquire) || isOperating.load(std::memory_order_acquire)) {
        return;
    }

    const VkExtent2D extent = swapchainManager.getExtent();
    const uint32_t targetWidth = windowState.width.load(std::memory_order_acquire);
    const uint32_t targetHeight = windowState.height.load(std::memory_order_acquire);
    if (targetWidth >= static_cast<uint32_t>(renderconfig::MinSwapchainExtent) &&
        targetHeight >= static_cast<uint32_t>(renderconfig::MinSwapchainExtent) &&
        (extent.width != targetWidth || extent.height != targetHeight)) {
        if (!recreateSwapChain()) {
            return;
        }
    }

    // Wait for the current slot's fences (only those signaled this lap). This
    // is the safe replacement for the unconditional compute-fence wait that
    // caused the permanent hang on submit failure / skipped compute.
    frameSync.waitForSlot();
    const uint32_t frameIndex = frameSync.getCurrentFrameIndex();

    FrameState frameState{};
    frameState.frameIndex = frameIndex;
    uint32_t imageIndex = 0;
    const FrameStageResult acquireResult = swapchainStage.acquireFrameImage(imageIndex);
    if (acquireResult == FrameStageResult::RecreateSwapchain) {
        std::cerr << "[DRAW] acquire recreate frameCall=" << callIndex << std::endl;
        (void)recreateSwapChain();
        return;
    }
    if (acquireResult != FrameStageResult::Continue) {
        std::cerr << "[DRAW] acquire fail result=" << static_cast<int>(acquireResult)
                  << " frameCall=" << callIndex << std::endl;
        return;
    }

    std::vector<std::string> timingLines = buildFrameTimingLines(frameState.frameIndex);
    frameUpdateStage.processPicking(frameState.frameIndex);

    frameState.extent = swapchainManager.getExtent();
    frameState.imageIndex = imageIndex;
    frameState.sceneView = cameraController.buildSceneView(frameState.extent);
    frameState.flags = flags;
    frameUpdateStage.updateFrameState(frameState.frameIndex, frameState.sceneView);

    // Stages only record work; they never touch sync. FrameController commits
    // the whole frame atomically via submitFrame below, so no stage can leave
    // a fence reset-without-signal.
    const FrameComputeCollection computeCollection = frameComputeStage.collect(frameState.frameIndex, computePasses);
    if (!handleStageResult(computeCollection.result)) {
        std::cerr << "[DRAW] compute stage fail result=" << static_cast<int>(computeCollection.result)
                  << " frameCall=" << callIndex << std::endl;
        return;
    }

    FrameSyncState syncState{};
    syncState.waitForComputeSemaphore = computeCollection.waitForComputeSemaphore;
    syncState.insertComputeToGraphicsBarrier = computeCollection.insertComputeToGraphicsBarrier;
    syncState.waitDstStageMask = computeCollection.waitDstStageMask;

    updateTimingOverlay(timingLines, frameState.flags);

    const FrameGraphicsCollection graphicsCollection = frameGraphicsStage.collect(frameState, syncState);
    if (!handleStageResult(graphicsCollection.result)) {
        std::cerr << "[DRAW] graphics stage fail result=" << static_cast<int>(graphicsCollection.result)
                  << " frameCall=" << callIndex << std::endl;
        return;
    }

    // Single atomic commit. submitFrame resets + submits both fences together
    // and records which were signaled so the next waitForSlot knows the truth.
    FrameSync::FrameSubmission submission{};
    submission.computeCommandBuffer = computeCollection.commandBuffer;
    submission.graphicsCommandBuffer = graphicsCollection.commandBuffer;
    submission.waitForComputeSemaphore = computeCollection.waitForComputeSemaphore;
    submission.insertComputeToGraphicsBarrier = computeCollection.insertComputeToGraphicsBarrier;
    submission.computeWaitDstStageMask = computeCollection.waitDstStageMask;

    const VkResult submitResult = frameSync.submitFrame(
        vulkanDevice.getComputeQueue(),
        vulkanDevice.getGraphicsQueue(),
        swapchainManager.getSwapChain(),
        frameState.imageIndex,
        submission);

    if (submitResult != VK_SUCCESS) {
        std::cerr << "[DRAW] submitResult=" << static_cast<int>(submitResult)
                  << " frameCall=" << callIndex << std::endl;
        if (submitResult == VK_ERROR_DEVICE_LOST) {
            return;
        }
        (void)recreateSwapChain();
        return;
    }

    const FrameStageResult presentResult = swapchainStage.presentFrame(frameState.imageIndex);
    if (presentResult == FrameStageResult::RecreateSwapchain) {
        std::cerr << "[DRAW] present recreate frameCall=" << callIndex << std::endl;
        (void)recreateSwapChain();
        return;
    }
    if (presentResult != FrameStageResult::Continue) {
        std::cerr << "[DRAW] present fail result=" << static_cast<int>(presentResult)
                  << " frameCall=" << callIndex << std::endl;
        return;
    }

    frameSync.advanceFrame();
}

std::vector<std::string> FrameController::buildFrameTimingLines(uint32_t frameIndex) {
    GpuTimingStats gpuTiming{};
    const GpuTimingStats* graphicsTiming = nullptr;
    if (sceneRenderer.getGpuTimingStats(frameIndex, gpuTiming)) {
        graphicsTiming = &gpuTiming;
    }

    return frameStats.buildTimingLines(graphicsTiming, std::nullopt);
}

void FrameController::updateTimingOverlay(std::vector<std::string>& timingLines, const render::RenderFlags& flags) {
    if (flags.drawTimingOverlay) {
        frameStats.capitalizeLines(timingLines);
        sceneRenderer.setTimingOverlayLines(timingLines);
        return;
    }

    sceneRenderer.setTimingOverlayLines({});
}

bool FrameController::handleStageResult(FrameStageResult result) {
    switch (result) {
    case FrameStageResult::Continue:
        return true;
    case FrameStageResult::SkipFrame:
        return false;
    case FrameStageResult::RecreateSwapchain:
        (void)recreateSwapChain();
        return false;
    case FrameStageResult::Fatal:
        return false;
    }

    return false;
}
