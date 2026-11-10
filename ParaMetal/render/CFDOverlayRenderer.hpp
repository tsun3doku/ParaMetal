#pragma once

#include "renderers/HeatBoxRenderer.hpp"
#include "renderers/StreamlineRenderer.hpp"
#include "runtime/CFDDisplayController.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

class MemoryAllocator;
class UniformBufferManager;
class VulkanDevice;
class CommandPool;

namespace render {

class CFDOverlayRenderer {
public:
    CFDOverlayRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~CFDOverlayRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void apply(uint64_t socketKey, const CFDDisplayController::Config& config);
    void remove(uint64_t socketKey);
    void renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    const std::vector<StreamlineRenderer::RenderBinding>& getStreamlineBindings() const { return streamlineBindings; }
    bool isVelocityPaletteVisible() const { return velocityPaletteVisible; }
    void cleanup();

private:
    void rebuildBindings();

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    MemoryAllocator& memoryAllocator;
    std::unique_ptr<HeatBoxRenderer> boxRenderer;
    std::unordered_map<uint64_t, CFDDisplayController::Config> configsBySocket;
    std::vector<StreamlineRenderer::RenderBinding> streamlineBindings;
    std::vector<HeatBoxRenderer::RenderBinding> boxBindings;
    uint32_t maxFramesInFlight = 0;
    bool initialized = false;
    bool velocityPaletteVisible = false;
};

}
