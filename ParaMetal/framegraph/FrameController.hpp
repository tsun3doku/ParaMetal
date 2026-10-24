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
class ModelRegistry;
class UniformBufferManager;
class ModelSelection;
class GizmoController;
class NavigationGizmoController;
class WireframeRenderer;
class VkFrameGraphBackend;
class VkFrameGraphRuntime;
class InputController;
class LightingSystem;
class MaterialSystem;

struct FrameControllerServices {
    ModelRegistry& modelRegistry;
    UniformBufferManager& uniformBufferManager;
    ModelSelection& modelSelection;
    GizmoController& gizmoController;
    NavigationGizmoController& navigationGizmoController;
    WireframeRenderer& wireframeRenderer;
    InputController& inputController;
    LightingSystem& lightingSystem;
    MaterialSystem& materialSystem;
};

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
        CameraController& cameraController,
        FrameControllerServices services,
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
    NavigationGizmoController& navigationGizmoController;
    std::atomic<bool>& isOperating;
    std::atomic<bool>& isShuttingDown;

    FrameUpdateStage frameUpdateStage;
    FrameGraphicsStage frameGraphicsStage;

    std::vector<std::string> buildFrameTimingLines(uint32_t frameIndex, std::optional<float> computeGpuMs);
    void updateTimingOverlay(std::vector<std::string>& timingLines, const render::RenderFlags& flags);

    VulkanDevice& vulkanDevice;
    VkFrameGraphRuntime& frameGraphRuntime;
};

