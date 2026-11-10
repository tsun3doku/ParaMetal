#include "CFDOverlayRenderer.hpp"

#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace render {

CFDOverlayRenderer::CFDOverlayRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      commandPool(commandPool),
      memoryAllocator(allocator),
      boxRenderer(std::make_unique<HeatBoxRenderer>(device, allocator, uniformBufferManager)) {
}

CFDOverlayRenderer::~CFDOverlayRenderer() {
    cleanup();
}

void CFDOverlayRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t updatedMaxFramesInFlight) {
    if (initialized) {
        return;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    if (boxRenderer) {
        boxRenderer->initialize(renderPass, subpass, maxFramesInFlight);
    }
    initialized = true;
    rebuildBindings();
}

void CFDOverlayRenderer::apply(uint64_t socketKey, const CFDDisplayController::Config& config) {
    if (socketKey == 0 || !config.anyVisible() || !config.isValid()) {
        remove(socketKey);
        return;
    }

    configsBySocket[socketKey] = config;
    rebuildBindings();
}

void CFDOverlayRenderer::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configsBySocket.erase(socketKey);
    rebuildBindings();
}

void CFDOverlayRenderer::rebuildBindings() {
    streamlineBindings.clear();
    boxBindings.clear();

    velocityPaletteVisible = false;

    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.active) continue;

        if (config.showVelocityPalette) {
            velocityPaletteVisible = true;
        }

        if (config.showStreamlines &&
            config.streamlineVertexBuffer != VK_NULL_HANDLE &&
            config.streamlineVertexCount != 0) {
            StreamlineRenderer::RenderBinding binding{};
            binding.bindingKey = socketKey;
            binding.vertexBuffer = config.streamlineVertexBuffer;
            binding.vertexBufferOffset = config.streamlineVertexBufferOffset;
            binding.vertexCount = config.streamlineVertexCount;

            streamlineBindings.push_back(binding);
        }

        if (config.showDomainBox && config.hasGlobalDomain) {
            HeatBoxRenderer::RenderBinding binding{};
            binding.corners = config.globalDomainCorners;

            boxBindings.push_back(binding);
        }
    }
}

void CFDOverlayRenderer::renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!initialized) {
        return;
    }

    if (boxRenderer) {
        boxRenderer->render(commandBuffer, frameIndex, boxBindings);
    }
}

void CFDOverlayRenderer::cleanup() {
    configsBySocket.clear();
    streamlineBindings.clear();
    boxBindings.clear();
    maxFramesInFlight = 0;
    initialized = false;
    velocityPaletteVisible = false;
    if (boxRenderer) {
        boxRenderer->cleanup();
    }
}

} // namespace render
