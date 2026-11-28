#include "OverlayPass.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "framegraph/FrameGraphPasses.hpp"
#include "GeometryPass.hpp"
#include "HeatOverlayRenderer.hpp"
#include "CFDOverlayRenderer.hpp"
#include "PointOverlayRenderer.hpp"
#include "VoronoiOverlayRenderer.hpp"
#include "scene/GizmoController.hpp"
#include "renderers/GizmoRenderer.hpp"
#include "renderers/NavigationGizmoRenderer.hpp"
#include "renderers/ScreenTextRenderer.hpp"
#include "scene/NavigationGizmoController.hpp"
#include "spatial/Grid.hpp"
#include "scene/ModelSelection.hpp"
#include "scene/SceneView.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "util/Structs.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "framegraph/VkFrameGraphRuntime.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "renderers/WireframeRenderer.hpp"
#include "renderers/TimingRenderer.hpp"
#include "renderers/IntrinsicRenderer.hpp"
#include "renderers/OutlineRenderer.hpp"
#include "renderers/SubselectionRenderer.hpp"
#include "RenderConfig.hpp"

namespace render {

OverlayPass::OverlayPass(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    VkFrameGraphRuntime& runtime,
    ModelRegistry& resources,
    UniformBufferManager& ubo,
    GeometryPass& geometry,
    HeatOverlayRenderer& heatOverlayRenderer,
    CFDOverlayRenderer& cfdOverlayRenderer,
    VoronoiOverlayRenderer& voronoiOverlayRenderer,
    IntrinsicRenderer& intrinsicRenderer,
    GizmoRenderer& gizmo,
    ScreenTextRenderer& screenText,
    NavigationGizmoRenderer& navigationGizmo,
    uint32_t framesInFlight,
    CommandPool& pool,
    framegraph::PassId passId,
    framegraph::ResourceId depthResolveId,
    framegraph::ResourceId depthMsaaId)
    : geometryPass(geometry),
      heatOverlayRenderer(heatOverlayRenderer),
      cfdOverlayRenderer(cfdOverlayRenderer),
      voronoiOverlayRenderer(voronoiOverlayRenderer),
      intrinsicRenderer(intrinsicRenderer),
      gizmoRenderer(gizmo),
      screenTextRenderer(screenText),
      navigationGizmoRenderer(navigationGizmo),
      vulkanDevice(device),
      memoryAllocator(allocator),
      frameGraphRuntime(runtime),
      modelRegistry(resources),
      uniformBufferManager(ubo),
      renderCommandPool(pool),
      maxFramesInFlight(framesInFlight),
      passId(passId),
      depthResolveId(depthResolveId),
      depthMsaaId(depthMsaaId) {
}

OverlayPass::~OverlayPass() = default;

const char* OverlayPass::name() const {
    return framegraph::passes::Overlay.data();
}

bool OverlayPass::create() {
    ready = false;
    destroy();
    outlineRenderer = std::make_unique<OutlineRenderer>(vulkanDevice);
    if (!outlineRenderer || !outlineRenderer->initialize(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight)) {
        std::cerr << "[OverlayPass] Failed to create outline renderer" << std::endl;
        destroy();
        return false;
    }

    if (!intrinsicRenderer.initializeOverlay(frameGraphRuntime.getRenderPass(), maxFramesInFlight, framegraph::toIndex(passId))) {
        std::cerr << "[OverlayPass] Failed to create intrinsic renderer" << std::endl;
        destroy();
        return false;
    }

    timingOverlay = std::make_unique<TimingRenderer>(screenTextRenderer);
    if (!timingOverlay) {
        std::cerr << "[OverlayPass] Failed to create timing overlay renderer" << std::endl;
        destroy();
        return false;
    }

    heatOverlayRenderer.initializeOverlay(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);
    cfdOverlayRenderer.initialize(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);
    voronoiOverlayRenderer.initializeOverlay(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);

    streamlineRenderer = std::make_unique<StreamlineRenderer>(
        vulkanDevice, memoryAllocator, uniformBufferManager, renderCommandPool);
    if (!streamlineRenderer) {
        std::cerr << "[OverlayPass] Failed to create streamline renderer" << std::endl;
        destroy();
        return false;
    }
    streamlineRenderer->initialize(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);

    sliceRenderer = std::make_unique<CFDSliceRenderer>(
        vulkanDevice, memoryAllocator, uniformBufferManager, renderCommandPool);
    if (!sliceRenderer) {
        std::cerr << "[OverlayPass] Failed to create velocity slice renderer" << std::endl;
        destroy();
        return false;
    }
    sliceRenderer->initialize(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);

    pointOverlayRenderer = std::make_unique<PointOverlayRenderer>(vulkanDevice, uniformBufferManager);
    if (!pointOverlayRenderer) {
        std::cerr << "[OverlayPass] Failed to create point overlay renderer" << std::endl;
        destroy();
        return false;
    }

    pointOverlayRenderer->initialize(frameGraphRuntime.getRenderPass(), framegraph::toIndex(passId), maxFramesInFlight);

    subselectionRenderer = std::make_unique<SubselectionRenderer>(
        vulkanDevice,
        memoryAllocator,
        uniformBufferManager);
    if (!subselectionRenderer) {
        std::cerr << "[OverlayPass] Failed to create subselection renderer" << std::endl;
        destroy();
        return false;
    }
    subselectionRenderer->initialize(
        frameGraphRuntime.getRenderPass(),
        framegraph::toIndex(passId),
        maxFramesInFlight);

    gridRenderer = std::make_unique<GridRenderer>(
        vulkanDevice,
        memoryAllocator,
        uniformBufferManager,
        maxFramesInFlight,
        frameGraphRuntime.getRenderPass(),
        framegraph::toIndex(passId),
        renderCommandPool);
    if (!gridRenderer) {
        std::cerr << "[OverlayPass] Failed to create grid renderer" << std::endl;
        destroy();
        return false;
    }
    ready = true;
    return true;
}

void OverlayPass::resize(VkExtent2D extent) {
    (void)extent;
}

void OverlayPass::updateDescriptors() {
    if (!ready) {
        return;
    }
    if (outlineRenderer) {
        outlineRenderer->updateDescriptors(
            frameGraphRuntime.getResourceStencilSamplerViews(depthMsaaId));
    }
}

void OverlayPass::setTimingOverlayLines(const std::vector<std::string>& lines) {
    if (timingOverlay) {
        timingOverlay->setLines(lines);
    }
}

void OverlayPass::updateGrid(
    uint32_t frameIndex,
    const SceneView& sceneView,
    const glm::vec3& sceneExtent) {
    if (gridRenderer) gridRenderer->update(frameIndex, sceneView, sceneExtent);
}

void OverlayPass::setWorldUnit(units::LengthUnit unit) {
    if (gridRenderer) gridRenderer->setWorldUnit(unit);
}

void OverlayPass::setVelocityPalette(uint32_t palette) {
    velocityPalette = palette;
}

void OverlayPass::setVelocityPaletteRange(float minimum, float maximum) {
    if (!(std::isfinite(minimum) && std::isfinite(maximum) && maximum > minimum)) {
        return;
    }
    velocityPaletteMinimum = minimum;
    velocityPaletteMaximum = maximum;
}

void OverlayPass::setStreamlineCull(float worldDistance) {
    if (!std::isfinite(worldDistance) || worldDistance < 0.0f) {
        return;
    }
    streamlineCullDistance = worldDistance;
}

PointOverlayRenderer* OverlayPass::getPointOverlayRenderer() const {
    return pointOverlayRenderer.get();
}

SubselectionRenderer* OverlayPass::getSubselectionRenderer() const {
    return subselectionRenderer.get();
}

void OverlayPass::record(const FrameContext& context, const SceneView& view, const RenderFlags& flags, RenderServices& services) {
    if (!ready) {
        return;
    }
    if (!services.modelSelection || !services.gizmoController || !services.wireframeRenderer) {
        return;
    }

    ModelSelection& modelSelection = *services.modelSelection;
    GizmoController& gizmoController = *services.gizmoController;
    WireframeRenderer& wireframeRenderer = *services.wireframeRenderer;
    VkCommandBuffer commandBuffer = context.commandBuffer;
    const uint32_t currentFrame = context.currentFrame;
    screenTextRenderer.beginFrame(currentFrame);
    const VkExtent2D extent = context.extent;
    if (sliceRenderer) {
        sliceRenderer->render(
            commandBuffer,
            currentFrame,
            context.extent,
            cfdOverlayRenderer.getSliceBindings(),
            velocityPalette,
            velocityPaletteMinimum,
            velocityPaletteMaximum);
    }
    heatOverlayRenderer.renderOverlay(commandBuffer, currentFrame, extent, view.proj * view.view);
    cfdOverlayRenderer.renderOverlay(commandBuffer, currentFrame);
    if (streamlineRenderer) {
        streamlineBindings = cfdOverlayRenderer.getStreamlineBindings();
        for (auto& binding : streamlineBindings) {
            binding.palette = velocityPalette;
            binding.minSpeed = velocityPaletteMinimum;
            binding.maxSpeed = velocityPaletteMaximum;
        }
        streamlineRenderer->render(commandBuffer, currentFrame, streamlineBindings, streamlineCullDistance);
    }
    voronoiOverlayRenderer.renderPoints(commandBuffer, currentFrame, extent);
    if (pointOverlayRenderer) {
        pointOverlayRenderer->render(commandBuffer, currentFrame, extent);
    }

    intrinsicRenderer.renderOverlay(commandBuffer, currentFrame);

    if (outlineRenderer && flags.wireframeMode == 0) {
        outlineRenderer->render(commandBuffer, currentFrame, modelSelection);
    }

    if (flags.drawGrid && gridRenderer && currentFrame < gridRenderer->getGridDescriptorSets().size()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, gridRenderer->getGridPipeline());
        vkCmdBindDescriptorSets(
            commandBuffer, 
            VK_PIPELINE_BIND_POINT_GRAPHICS, 
            gridRenderer->getGridPipelineLayout(), 
            0, 
            1, 
            &gridRenderer->getGridDescriptorSets()[currentFrame], 
            0, 
            nullptr);
        vkCmdDraw(commandBuffer, gridRenderer->vertexCount, 1, 0, 0);
        if (flags.drawAxisLabels) {
            gridRenderer->renderLabels(commandBuffer, currentFrame);
        }
    }

    if (flags.wireframeMode > 0) {
        VkDescriptorSet geometryDescriptorSet = geometryPass.getDescriptorSet(currentFrame);
        if (geometryDescriptorSet != VK_NULL_HANDLE) {
            wireframeRenderer.bindPipeline(commandBuffer);
            for (uint32_t modelId : modelRegistry.getRenderableModelIds()) {
                VkBuffer vertexBuffer = VK_NULL_HANDLE;
                VkDeviceSize vertexBufferOffset = 0;
                VkBuffer indexBuffer = VK_NULL_HANDLE;
                VkDeviceSize indexBufferOffset = 0;
                uint32_t indexCount = 0;
                glm::mat4 modelMatrix{1.0f};
                if (!modelRegistry.tryGetRenderGeometry(
                        modelId, vertexBuffer, vertexBufferOffset,
                        indexBuffer, indexBufferOffset, indexCount) ||
                    !modelRegistry.tryGetModelMatrix(modelId, modelMatrix)) {
                    continue;
                }
                const glm::vec3 wireframeColor =
                    modelSelection.isModelSelected(modelId)
                    ? renderconfig::SelectedWireframeColor
                    : renderconfig::WireframeColor;
                wireframeRenderer.renderModel(
                    commandBuffer, geometryDescriptorSet, modelMatrix,
                    vertexBuffer, vertexBufferOffset,
                    indexBuffer, indexBufferOffset, indexCount, wireframeColor);
            }
        }
    }

    if (flags.wireframeMode == 1) {
        intrinsicRenderer.renderXRayWireframe(
            commandBuffer, currentFrame, modelSelection);
    }

    if (subselectionRenderer) {
        subselectionRenderer->render(
            commandBuffer,
            currentFrame,
            extent,
            modelRegistry);
    }

    if (flags.drawTimingOverlay && timingOverlay) {
        timingOverlay->render(commandBuffer, currentFrame, extent);
    }

    if (modelSelection.getSelected()) {
        VkClearAttachment clearAttachment{};
        clearAttachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clearAttachment.clearValue.depthStencil = { 1.0f, 0 };

        VkClearRect clearRect{};
        clearRect.rect.offset = { 0, 0 };
        clearRect.rect.extent = extent;
        clearRect.baseArrayLayer = 0;
        clearRect.layerCount = 1;

        vkCmdClearAttachments(commandBuffer, 1, &clearAttachment, 1, &clearRect);

        const glm::vec3 gizmoPosition = gizmoController.calculateGizmoPosition(modelRegistry, modelSelection);
        const float gizmoScale = gizmoRenderer.calculateGizmoScale(modelRegistry, modelSelection);
        gizmoRenderer.render(commandBuffer, gizmoPosition, extent, gizmoScale, view, gizmoController);
    }

    if (flags.drawNavigationCube && services.navigationGizmoController) {
        navigationGizmoRenderer.render(commandBuffer, services.navigationGizmoController->getRenderData());
    }
}

void OverlayPass::destroy() {
    ready = false;
    if (outlineRenderer) {
        outlineRenderer->cleanup();
        outlineRenderer.reset();
    }
    if (pointOverlayRenderer) {
        pointOverlayRenderer->cleanup();
        pointOverlayRenderer.reset();
    }
    if (subselectionRenderer) {
        subselectionRenderer->cleanup();
        subselectionRenderer.reset();
    }
    if (timingOverlay) {
        timingOverlay->cleanup();
        timingOverlay.reset();
    }
    if (gridRenderer) {
        gridRenderer->cleanup();
        gridRenderer.reset();
    }
    if (streamlineRenderer) {
        streamlineRenderer->cleanup();
        streamlineRenderer.reset();
    }
    if (sliceRenderer) {
        sliceRenderer->cleanup();
        sliceRenderer.reset();
    }
    streamlineBindings.clear();
}

} // namespace render
