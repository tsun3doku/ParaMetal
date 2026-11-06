#include "RenderRuntime.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "framegraph/FrameController.hpp"
#include "framegraph/FrameGraph.hpp"
#include "framegraph/FrameGraphPasses.hpp"
#include "framegraph/FrameGraphResources.hpp"
#include "framegraph/FrameGraphVkTypes.hpp"
#include "framegraph/FrameSync.hpp"
#include "scene/IBLSystem.hpp"
#include "scene/LightingSystem.hpp"
#include "MainRenderGraph.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "scene/MaterialSystem.hpp"
#include "scene/ModelSelection.hpp"
#include "RenderConfig.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "SceneRenderer.hpp"
#include "app/ViewportTarget.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "framegraph/VkFrameGraphBackend.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "renderers/WireframeRenderer.hpp"

#include <iostream>

RenderRuntime::RenderRuntime(
    const WindowRuntimeState& windowState,
    VulkanDevice& vulkanDevice,
    ViewportTarget& viewportTarget,
    CommandPool& renderCommandPool,
    FrameSync& frameSync,
    std::atomic<bool>& isOperating,
    std::atomic<bool>& isShuttingDown)
    : windowState(windowState),
      vulkanDevice(vulkanDevice),
      viewportTarget(viewportTarget),
      renderCommandPool(renderCommandPool),
      frameSync(frameSync),
      isOperating(isOperating),
      isShuttingDown(isShuttingDown) {
}

RenderRuntime::~RenderRuntime() = default;

bool RenderRuntime::initializeBase(VkFormat swapChainFormat, VkExtent2D extent, MemoryAllocator& allocator, ModelRegistry& modelRegistry, UniformBufferManager& ubo, IBLSystem& iblSystem) {
    if (swapChainFormat == VK_FORMAT_UNDEFINED || extent.width == 0 || extent.height == 0) {
        return false;
    }

    frameGraph = std::make_unique<FrameGraph>();
    if (!frameGraph) {
        return false;
    }
    MainRenderGraph::buildMainRenderGraph(*frameGraph);

    if (!frameGraph->compile(framegraph::vk::toFrameGraphFormat(swapChainFormat), framegraph::vk::toFrameGraphExtent(extent))) {
        return false;
    }

    frameGraphBackend = std::make_unique<VkFrameGraphBackend>(vulkanDevice, allocator);
    if (!frameGraphBackend) {
        return false;
    }
    if (!frameGraphBackend->rebuild(frameGraph->getFrameGraphResult(), viewportTarget.getImageViews(), extent, renderconfig::MaxFramesInFlight)) {
        return false;
    }

    sceneRenderer = std::make_unique<SceneRenderer>(
        vulkanDevice,
        allocator,
        *frameGraph,
        frameGraphBackend->getRuntime(),
        modelRegistry,
        ubo,
        renderconfig::MaxFramesInFlight,
        renderCommandPool,
        iblSystem);
    if (!sceneRenderer) {
        return false;
    }
    if (!sceneRenderer->isReady()) {
        return false;
    }
    sceneRenderer->updateDescriptorSets();

    modelSelection = std::make_unique<ModelSelection>(
        vulkanDevice,
        frameGraphBackend->getRuntime(),
        modelRegistry,
        frameGraph->getResourceId(framegraph::resources::PickID));
    if (!modelSelection) {
        return false;
    }
    if (!modelSelection->isInitialized()) {
        return false;
    }

    wireframeRenderer = std::make_unique<WireframeRenderer>(
        vulkanDevice,
        sceneRenderer->getGbufferDescriptorSetLayout(),
        frameGraphBackend->getRuntime().getRenderPass(),
        framegraph::toIndex(frameGraph->getPassId(framegraph::passes::Overlay)));
    if (!wireframeRenderer) {
        return false;
    }

    return true;
}

bool RenderRuntime::initializeFrameController(SceneContext& scene, InteractionContext& interaction) {
    if (!modelSelection || !wireframeRenderer || !frameGraph || !frameGraphBackend || !sceneRenderer) {
        return false;
    }

    frameController = std::make_unique<FrameController>(
        windowState,
        vulkanDevice,
        viewportTarget,
        *frameGraphBackend,
        *sceneRenderer,
        frameSync,
        frameStats,
        scene,
        interaction,
        *modelSelection,
        *wireframeRenderer,
        isOperating,
        isShuttingDown);

    return frameController != nullptr;
}

bool RenderRuntime::renderFrame(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const render::RenderFlags& flags) {
    if (frameController) {
        return frameController->drawFrame(commandBuffer, frameIndex, flags);
    }
    return false;
}

VulkanDevice& RenderRuntime::getVulkanDevice() {
    return vulkanDevice;
}

const VulkanDevice& RenderRuntime::getVulkanDevice() const {
    return vulkanDevice;
}

FrameStats& RenderRuntime::getFrameStats() {
    return frameStats;
}

const FrameStats& RenderRuntime::getFrameStats() const {
    return frameStats;
}

bool RenderRuntime::updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent) {
    if (!frameGraph || !frameGraphBackend || !sceneRenderer) {
        return false;
    }

    vkDeviceWaitIdle(vulkanDevice.getDevice());
    frameGraphBackend->cleanup(renderconfig::MaxFramesInFlight);
    if (!viewportTarget.update(image, format, extent)) {
        return false;
    }
    if (!frameGraph->compile(
            framegraph::vk::toFrameGraphFormat(format),
            framegraph::vk::toFrameGraphExtent(extent))) {
        return false;
    }
    if (!frameGraphBackend->rebuild(
            frameGraph->getFrameGraphResult(),
            viewportTarget.getImageViews(),
            extent,
            renderconfig::MaxFramesInFlight)) {
        return false;
    }
    sceneRenderer->resize(extent);
    sceneRenderer->updateDescriptorSets();
    return true;
}

void RenderRuntime::cleanup() {
    if (modelSelection) {
        modelSelection->cleanup();
    }
    if (frameGraphBackend) {
        frameGraphBackend->cleanup(renderconfig::MaxFramesInFlight);
    }
    if (sceneRenderer) {
        sceneRenderer->cleanup();
    }
    if (wireframeRenderer) {
        wireframeRenderer->cleanup();
    }

    frameController.reset();
    wireframeRenderer.reset();
    modelSelection.reset();
    sceneRenderer.reset();
    frameGraphBackend.reset();
    frameGraph.reset();
}

FrameGraph& RenderRuntime::getFrameGraph() {
    return *frameGraph;
}

const FrameGraph& RenderRuntime::getFrameGraph() const {
    return *frameGraph;
}

VkFrameGraphRuntime& RenderRuntime::getFrameGraphRuntime() {
    return frameGraphBackend->getRuntime();
}

const VkFrameGraphRuntime& RenderRuntime::getFrameGraphRuntime() const {
    return frameGraphBackend->getRuntime();
}

SceneRenderer& RenderRuntime::getSceneRenderer() {
    return *sceneRenderer;
}

const SceneRenderer& RenderRuntime::getSceneRenderer() const {
    return *sceneRenderer;
}

ModelSelection& RenderRuntime::getModelSelection() {
    return *modelSelection;
}

const ModelSelection& RenderRuntime::getModelSelection() const {
    return *modelSelection;
}

WireframeRenderer& RenderRuntime::getWireframeRenderer() {
    return *wireframeRenderer;
}

const WireframeRenderer& RenderRuntime::getWireframeRenderer() const {
    return *wireframeRenderer;
}

