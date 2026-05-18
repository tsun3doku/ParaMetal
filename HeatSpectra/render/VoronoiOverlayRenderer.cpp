#include "VoronoiOverlayRenderer.hpp"

#include <glm/mat4x4.hpp>

#include "renderers/PointRenderer.hpp"
#include "renderers/VoronoiRenderer.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace render {

VoronoiOverlayRenderer::VoronoiOverlayRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& renderCommandPool)
    : voronoiRenderer(std::make_unique<VoronoiRenderer>(device, allocator, uniformBufferManager, renderCommandPool)),
      pointRenderer(std::make_unique<PointRenderer>(device, uniformBufferManager)) {
}

VoronoiOverlayRenderer::~VoronoiOverlayRenderer() {
    cleanup();
}

void VoronoiOverlayRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight) {
    if (!voronoiRenderer || !pointRenderer || initialized) {
        return;
    }

    voronoiRenderer->initialize(renderPass, maxFramesInFlight);
    pointRenderer->initialize(renderPass, subpass, maxFramesInFlight);
    initialized = true;
}

void VoronoiOverlayRenderer::apply(uint64_t socketKey, const VoronoiDisplayController::Config& config) {
    if (socketKey == 0 || !config.anyVisible() || !config.isValid()) {
        remove(socketKey);
        return;
    }

    configsBySocket[socketKey] = config;
}

void VoronoiOverlayRenderer::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configsBySocket.erase(socketKey);
}

void VoronoiOverlayRenderer::renderSurface(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!initialized || !voronoiRenderer) {
        return;
    }

    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (!config.showVoronoi) {
            continue;
        }

        for (size_t i = 0; i < config.modelRuntimeModelIds.size(); ++i) {
            voronoiRenderer->updateDescriptors(
                frameIndex,
                static_cast<uint32_t>(config.modelIntrinsicVertexCounts[i]),
                config.seedPositionBuffer,
                config.seedPositionBufferOffset,
                config.neighborIndicesBuffer,
                config.neighborIndicesBufferOffset,
                config.modelSupportingHalfedgeViews[i],
                config.modelSupportingAngleViews[i],
                config.modelHalfedgeViews[i],
                config.modelEdgeViews[i],
                config.modelTriangleViews[i],
                config.modelLengthViews[i],
                config.modelInputHalfedgeViews[i],
                config.modelInputEdgeViews[i],
                config.modelInputTriangleViews[i],
                config.modelInputLengthViews[i],
                config.modelCandidateBuffers[i],
                config.modelCandidateBufferOffsets[i]);

            voronoiRenderer->render(
                commandBuffer,
                config.modelVertexBuffers[i],
                config.modelVertexBufferOffsets[i],
                config.modelIndexBuffers[i],
                config.modelIndexBufferOffsets[i],
                config.modelIndexCounts[i],
                frameIndex,
                config.modelMatrices[i]);
        }
    }
}

void VoronoiOverlayRenderer::renderPoints(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent) {
    if (!initialized || !pointRenderer) {
        return;
    }

    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (!config.showPoints || config.occupancyPointBuffer == VK_NULL_HANDLE || config.occupancyPointCount == 0) {
            continue;
        }

        pointRenderer->render(
            commandBuffer,
            frameIndex,
            config.occupancyPointBuffer,
            config.occupancyPointBufferOffset,
            config.occupancyPointCount,
            glm::mat4(1.0f),
            extent);
    }
}

void VoronoiOverlayRenderer::cleanup() {
    configsBySocket.clear();
    initialized = false;
    if (voronoiRenderer) {
        voronoiRenderer->cleanup();
    }
    if (pointRenderer) {
        pointRenderer->cleanup();
    }
}

} // namespace render
