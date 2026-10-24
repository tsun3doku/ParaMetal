#pragma once

#include <atomic>
#include <memory>

#include "NodeGraphContext.hpp"
#include "app/AppTypes.hpp"
#include "runtime/ModelDisplayController.hpp"
#include "runtime/PointDisplayController.hpp"
#include "runtime/RemeshDisplayController.hpp"
#include "runtime/HeatDisplayController.hpp"
#include "runtime/VoronoiDisplayController.hpp"
#include "app/ViewportTarget.hpp"
#include "render/RenderRuntime.hpp"

class FrameSync;
class SceneContext;
class VulkanCoreContext;
struct WindowRuntimeState;

class RenderContext {
public:
    ~RenderContext();

    bool initialize(
        VulkanCoreContext& core,
        SceneContext& scene,
        FrameSync& frameSync,
        WindowRuntimeState& windowState,
        const AppVulkanContext& vulkanContext,
        std::atomic<bool>& runtimeBusy,
        std::atomic<bool>& isShuttingDown);
    void shutdown();
    bool isInitialized() const;

    bool updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent);
    NodeGraphDisplayControllers displayControllers();
    RenderRuntime* runtime();
    const RenderRuntime* runtime() const;

private:
    const WindowRuntimeState* windowState = nullptr;
    ViewportTarget viewportTarget;
    std::unique_ptr<RenderRuntime> renderRuntime;
    std::unique_ptr<ModelDisplayController> modelDisplayControllerState;
    std::unique_ptr<PointDisplayController> pointDisplayControllerState;
    std::unique_ptr<RemeshDisplayController> remeshDisplayControllerState;
    std::unique_ptr<HeatDisplayController> heatDisplayControllerState;
    std::unique_ptr<VoronoiDisplayController> voronoiDisplayControllerState;
    bool initialized = false;
};
