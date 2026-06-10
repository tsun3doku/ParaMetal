#include "PointOverlayRenderer.hpp"

#include "renderers/PointRenderer.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace render {

PointOverlayRenderer::PointOverlayRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager)
    : vulkanDevice(device), uniformBufferManager(uniformBufferManager),
      pointRenderer(std::make_unique<PointRenderer>(device, uniformBufferManager)) {
}

PointOverlayRenderer::~PointOverlayRenderer() {
    cleanup();
}

void PointOverlayRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight) {
    if (!pointRenderer || initialized) {
        return;
    }

    this->maxFramesInFlight = maxFramesInFlight;
    pointRenderer->initialize(renderPass, subpass, maxFramesInFlight);
    initialized = true;
}

void PointOverlayRenderer::apply(uint64_t socketKey, const PointDisplayController::Config& config) {
    if (socketKey == 0 || !config.isValid()) {
        remove(socketKey);
        return;
    }
    configsBySocket[socketKey] = config;
}

void PointOverlayRenderer::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    configsBySocket.erase(socketKey);
}

void PointOverlayRenderer::render(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent) {
    if (!initialized || !pointRenderer) {
        return;
    }

    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (config.vertexBuffer == VK_NULL_HANDLE || config.pointCount == 0) {
            continue;
        }
        pointRenderer->render(
            commandBuffer,
            frameIndex,
            config.vertexBuffer,
            config.vertexBufferOffset,
            config.pointCount,
            config.modelMatrix,
            extent);
    }
}

void PointOverlayRenderer::cleanup() {
    if (pointRenderer) {
        pointRenderer->cleanup();
    }
    configsBySocket.clear();
    initialized = false;
}

} // namespace render
