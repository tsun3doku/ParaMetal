#include "FrameController.hpp"

#include "scene/CameraController.hpp"
#include "scene/NavigationGizmoController.hpp"
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
      navigationGizmoController(services.navigationGizmoController),
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
          services.navigationGizmoController,
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

    frameSync.waitForSlot();
    const uint32_t frameIndex = frameSync.getCurrentFrameIndex();

    FrameState frameState{};
    frameState.frameIndex = frameIndex;
    std::vector<std::string> timingLines;
    if (flags.drawTimingOverlay) {
        timingLines = buildFrameTimingLines(frameState.frameIndex);
    }
    frameUpdateStage.processPicking(frameState.frameIndex);

    frameState.extent = swapchainManager.getExtent();
    navigationGizmoController.setViewport(
        frameState.extent,
        windowState.devicePixelRatio.load(std::memory_order_acquire));
    frameState.sceneView = cameraController.buildSceneView(frameState.extent);
    frameState.flags = flags;
    frameUpdateStage.updateFrameState(frameState.frameIndex, frameState.sceneView);

    const FrameComputeCollection computeCollection = frameComputeStage.collect(frameState.frameIndex, computePasses);
    if (computeCollection.result != FrameStageResult::Continue) {
        std::cerr << "[DRAW] compute stage fail result=" << static_cast<int>(computeCollection.result)
                  << " frameCall=" << callIndex << std::endl;
        if (computeCollection.result == FrameStageResult::RecreateSwapchain) {
            (void)recreateSwapChain();
        } else if (computeCollection.result == FrameStageResult::Fatal) {
            isShuttingDown.store(true, std::memory_order_release);
        }
        return;
    }

    FrameSyncState syncState{};
    syncState.waitForComputeSemaphore = computeCollection.waitForComputeSemaphore;
    syncState.insertComputeToGraphicsBarrier = computeCollection.insertComputeToGraphicsBarrier;
    syncState.waitDstStageMask = computeCollection.waitDstStageMask;

    uint32_t imageIndex = 0;
    const FrameStageResult acquireResult = swapchainStage.acquireFrameImage(imageIndex);
    if (acquireResult == FrameStageResult::RecreateSwapchain) {
        std::cerr << "[DRAW] acquire recreate frameCall=" << callIndex << std::endl;
        (void)recreateSwapChain();
        return;
    }
    if (acquireResult != FrameStageResult::Continue) {
        std::cerr << "[DRAW] acquire fatal frameCall=" << callIndex << std::endl;
        isShuttingDown.store(true, std::memory_order_release);
        return;
    }
    frameState.imageIndex = imageIndex;

    updateTimingOverlay(timingLines, frameState.flags);

    const FrameGraphicsCollection graphicsCollection = frameGraphicsStage.collect(frameState, syncState);
    if (graphicsCollection.result != FrameStageResult::Continue) {
        std::cerr << "[DRAW] graphics stage fail result=" << static_cast<int>(graphicsCollection.result)
                  << " frameCall=" << callIndex << std::endl;
        if (graphicsCollection.result == FrameStageResult::RecreateSwapchain) {
            (void)recreateSwapChain();
        } else if (graphicsCollection.result == FrameStageResult::Fatal) {
            isShuttingDown.store(true, std::memory_order_release);
        }
        return;
    }

    FrameSync::FrameSubmission submission{};
    submission.computeCommandBuffer = computeCollection.commandBuffer;
    submission.graphicsCommandBuffer = graphicsCollection.commandBuffer;
    submission.waitForComputeSemaphore = computeCollection.waitForComputeSemaphore;
    submission.insertComputeToGraphicsBarrier = computeCollection.insertComputeToGraphicsBarrier;
    submission.computeWaitDstStageMask = computeCollection.waitDstStageMask;
    submission.externalWaitSemaphore = computeCollection.synchronization.waitSemaphore;
    submission.externalWaitValue = computeCollection.synchronization.waitValue;
    submission.externalSignalSemaphore = computeCollection.synchronization.signalSemaphore;
    submission.externalSignalValue = computeCollection.synchronization.signalValue;

    const VkResult submitResult = frameSync.submitFrame(
        vulkanDevice.getComputeQueue(),
        vulkanDevice.getGraphicsQueue(),
        swapchainManager.getSwapChain(),
        frameState.imageIndex,
        submission);

    if (submitResult != VK_SUCCESS) {
        std::cerr << "[DRAW] submitResult=" << static_cast<int>(submitResult) << " frameCall=" << callIndex << std::endl;
        if (submitResult == VK_ERROR_DEVICE_LOST) {
            isShuttingDown.store(true, std::memory_order_release);
            return;
        }
        (void)recreateSwapChain();
        return;
    }

    const FrameStageResult presentResult = swapchainStage.presentFrame(frameState.imageIndex);
    if (presentResult == FrameStageResult::RecreateSwapchain) {
        (void)recreateSwapChain();
        return;
    }
    if (presentResult != FrameStageResult::Continue) {
        isShuttingDown.store(true, std::memory_order_release);
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
