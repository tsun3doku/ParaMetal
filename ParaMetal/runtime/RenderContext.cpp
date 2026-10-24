#include "RenderContext.hpp"

#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "app/ViewportTarget.hpp"
#include "render/RenderRuntime.hpp"
#include "render/SceneRenderer.hpp"
#include "render/WindowRuntimeState.hpp"
#include "scene/IBLSystem.hpp"
#include "scene/LightingSystem.hpp"
#include "scene/MaterialSystem.hpp"

RenderContext::~RenderContext() = default;

bool RenderContext::initialize(VulkanCoreContext& core, SceneContext& scene, FrameSync& frameSync, WindowRuntimeState& windowState, const AppVulkanContext& vulkanContext,
    std::atomic<bool>& runtimeBusy, std::atomic<bool>& isShuttingDown) {
    if (initialized) {
        return true;
    }

    this->windowState = &windowState;

    auto* allocator = core.allocator();
    auto* commandPool = core.getRenderCommandPool();
    auto* modelRegistry = scene.modelRegistry();
    auto* uniformBufferManager = scene.uniformBufferManager();
    auto* iblSystem = scene.iblSystem();
    if (!allocator || !commandPool || !modelRegistry || !uniformBufferManager || !iblSystem) {
        return false;
    }

    viewportTarget.initialize(core.device());
    if (!viewportTarget.update(
            vulkanContext.viewportImage,
            vulkanContext.viewportFormat,
            vulkanContext.viewportExtent)) {
        return false;
    }

    const VkFormat viewportImageFormat = viewportTarget.getImageFormat();
    const VkExtent2D viewportExtent = viewportTarget.getExtent();

    renderRuntime = std::make_unique<RenderRuntime>(
        windowState,
        core.device(),
        viewportTarget,
        *commandPool,
        frameSync,
        scene.cameraController(),
        runtimeBusy,
        isShuttingDown);

    if (!renderRuntime->initializeBase(
        viewportImageFormat,
        viewportExtent,
        *allocator,
        *modelRegistry,
        *uniformBufferManager,
        *iblSystem)) {
        shutdown();
        return false;
    }

    pointDisplayControllerState = std::make_unique<PointDisplayController>();
    modelDisplayControllerState = std::make_unique<ModelDisplayController>();
    remeshDisplayControllerState = std::make_unique<RemeshDisplayController>();
    voronoiDisplayControllerState = std::make_unique<VoronoiDisplayController>();
    heatDisplayControllerState = std::make_unique<HeatDisplayController>();

    voronoiDisplayControllerState->setOverlayRenderer(renderRuntime->getSceneRenderer().getVoronoiOverlayRenderer());
    heatDisplayControllerState->setOverlayRenderer(renderRuntime->getSceneRenderer().getHeatOverlayRenderer());
    heatDisplayControllerState->setVolumeRenderer(renderRuntime->getSceneRenderer().getHeatVolumeRenderer());
    pointDisplayControllerState->setOverlayRenderer(renderRuntime->getSceneRenderer().getPointOverlayRenderer());
    modelDisplayControllerState->setModelRegistry(modelRegistry);
    remeshDisplayControllerState->setIntrinsicRenderer(renderRuntime->getSceneRenderer().getIntrinsicRenderer());

    initialized = true;
    return true;
}

void RenderContext::shutdown() {
    remeshDisplayControllerState.reset();
    modelDisplayControllerState.reset();
    pointDisplayControllerState.reset();
    voronoiDisplayControllerState.reset();
    heatDisplayControllerState.reset();

    if (renderRuntime) {
        renderRuntime->cleanup();
    }
    renderRuntime.reset();
    viewportTarget.cleanup();
    windowState = nullptr;
    initialized = false;
}

bool RenderContext::isInitialized() const {
    return initialized;
}

bool RenderContext::updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent) {
    return renderRuntime && renderRuntime->updateViewportTarget(image, format, extent);
}

NodeGraphDisplayControllers RenderContext::displayControllers() {
    return {
        *modelDisplayControllerState,
        *pointDisplayControllerState,
        *remeshDisplayControllerState,
        *heatDisplayControllerState,
        *voronoiDisplayControllerState};
}

RenderRuntime* RenderContext::runtime() {
    return renderRuntime.get();
}

const RenderRuntime* RenderContext::runtime() const {
    return renderRuntime.get();
}
