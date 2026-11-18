#include "CFDOverlayRenderer.hpp"

#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <algorithm>
#include <cmath>

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
    sliceBindings.clear();
    boxBindings.clear();

    velocityPaletteVisible = false;

    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.active) continue;

        if (config.showVelocityPalette || config.showSlice) {
            velocityPaletteVisible = true;
        }

        if (config.showSlice &&
            config.hasGlobalDomain &&
            std::isfinite(config.slicePosition) &&
            config.velocityBuffer != VK_NULL_HANDLE &&
            config.gradientBuffer != VK_NULL_HANDLE &&
            config.centroidBuffer != VK_NULL_HANDLE &&
            config.sliceGeometry != nullptr) {
            CFDSliceRenderer::RenderBinding binding{};
            binding.bindingKey = socketKey;
            binding.velocityBuffer = config.velocityBuffer;
            binding.velocityBufferOffset = config.velocityBufferOffset;
            binding.gradientBuffer = config.gradientBuffer;
            binding.gradientBufferOffset = config.gradientBufferOffset;
            binding.centroidBuffer = config.centroidBuffer;
            binding.centroidBufferOffset = config.centroidBufferOffset;
            binding.fluidNodeCount = config.fluidNodeCount;
            binding.domainMin = config.globalDomainCorners[0];
            binding.domainMax = config.globalDomainCorners[0];
            for (const glm::vec3& corner : config.globalDomainCorners) {
                binding.domainMin = glm::min(binding.domainMin, corner);
                binding.domainMax = glm::max(binding.domainMax, corner);
            }
            binding.planeAxis = static_cast<uint32_t>(std::clamp(config.sliceAxis, 0, 2));
            binding.planeFraction = std::clamp(config.slicePosition, 0.0f, 1.0f);
            binding.showWireframe = config.showSliceWireframe;
            binding.sliceGeometry = config.sliceGeometry;

            sliceBindings.push_back(binding);
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

        if (config.hasGlobalDomain) {
            HeatBoxRenderer::RenderBinding binding{};
            binding.corners = config.globalDomainCorners;
            binding.outline = config.showDomainBox;

            boxBindings.push_back(binding);
        }
    }
}

void CFDOverlayRenderer::renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!initialized) {
        return;
    }

    if (boxRenderer) {
        boxRenderer->render(commandBuffer, frameIndex, boxBindings, boundaryHighlight);
    }
}

void CFDOverlayRenderer::setBoundaryFaceHighlight(
    int selectedFace,
    int hoverFace,
    const std::array<glm::vec3, 8>& corners,
    bool valid) {
    boundaryHighlight.selectedFace = selectedFace;
    boundaryHighlight.hoverFace = hoverFace;
    boundaryHighlight.corners = corners;
    boundaryHighlight.boundsValid = valid;
}

void CFDOverlayRenderer::cleanup() {
    configsBySocket.clear();
    streamlineBindings.clear();
    sliceBindings.clear();
    boxBindings.clear();
    maxFramesInFlight = 0;
    initialized = false;
    velocityPaletteVisible = false;
    if (boxRenderer) {
        boxRenderer->cleanup();
    }
}

} // namespace render
