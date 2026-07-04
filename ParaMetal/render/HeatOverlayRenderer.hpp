#pragma once

#include "renderers/HeatSurfaceRenderer.hpp"
#include "renderers/HeatPaletteRenderer.hpp"
#include "renderers/VectorArrowRenderer.hpp"
#include "runtime/HeatDisplayController.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

class MemoryAllocator;
class UniformBufferManager;
class VulkanDevice;
class CommandPool;

namespace render {

class HeatOverlayRenderer {
public:
    HeatOverlayRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~HeatOverlayRenderer();

    void initializeSurface(VkRenderPass renderPass, uint32_t surfaceSubpass, uint32_t maxFramesInFlight);
    void initializeOverlay(VkRenderPass renderPass, uint32_t overlaySubpass, uint32_t maxFramesInFlight);
    void apply(uint64_t socketKey, const HeatDisplayController::Config& config);
    void remove(uint64_t socketKey);
    void renderSurface(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    void renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex);
    void renderScreen(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent);
    void cleanup();

private:
    void rebuildBindings();

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    MemoryAllocator& memoryAllocator;
    std::unique_ptr<HeatSurfaceRenderer> surfaceRenderer;
    std::unique_ptr<VectorArrowRenderer> vectorArrowRenderer;
    std::unique_ptr<HeatPaletteRenderer> heatPaletteRenderer;
    std::unordered_map<uint64_t, HeatDisplayController::Config> configsBySocket;
    std::vector<HeatSurfaceRenderer::SurfaceRenderBinding> surfaceBindings;
    std::vector<VectorArrowRenderer::VectorRenderBinding> fluxVectorBindings;
    uint32_t maxFramesInFlight = 0;
    bool surfaceInitialized = false;
    bool overlayInitialized = false;
};

}
