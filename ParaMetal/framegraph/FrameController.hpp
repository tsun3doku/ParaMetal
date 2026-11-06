#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "FrameGraphicsStage.hpp"
#include "FrameTypes.hpp"
#include "FrameUpdateStage.hpp"

struct WindowRuntimeState;
class VulkanDevice;
class ViewportTarget;
class SceneRenderer;
class FrameSync;
class FrameStats;
class CameraController;
class SceneContext;
class InteractionContext;
class ModelSelection;
class WireframeRenderer;
class VkFrameGraphBackend;
class VkFrameGraphRuntime;

class FrameController {
public:
    FrameController(
        const WindowRuntimeState& windowState,
        VulkanDevice& vulkanDevice,
        ViewportTarget& viewportTarget,
        VkFrameGraphBackend& frameGraphBackend,
        SceneRenderer& sceneRenderer,
        FrameSync& frameSync,
        FrameStats& frameStats,
        SceneContext& scene,
        InteractionContext& interaction,
        ModelSelection& modelSelection,
        WireframeRenderer& wireframeRenderer,
        std::atomic<bool>& isOperating,
        std::atomic<bool>& isShuttingDown);

    bool drawFrame(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const render::RenderFlags& flags);

private:
    const WindowRuntimeState& windowState;
    ViewportTarget& viewportTarget;
    SceneRenderer& sceneRenderer;
    FrameSync& frameSync;
    FrameStats& frameStats;
    CameraController& cameraController;
    InteractionContext& interaction;
    std::atomic<bool>& isOperating;
    std::atomic<bool>& isShuttingDown;

    FrameUpdateStage frameUpdateStage;
    FrameGraphicsStage frameGraphicsStage;

    std::vector<std::string> buildFrameTimingLines(uint32_t frameIndex, std::optional<float> computeGpuMs);
    void updateTimingOverlay(std::vector<std::string>& timingLines, const render::RenderFlags& flags);

    VulkanDevice& vulkanDevice;
    VkFrameGraphRuntime& frameGraphRuntime;
};
