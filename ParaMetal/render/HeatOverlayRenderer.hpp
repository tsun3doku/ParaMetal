#pragma once

#include "renderers/HeatBoxRenderer.hpp"
#include "renderers/HeatSurfaceRenderer.hpp"
#include "renderers/ProbeMarkerRenderer.hpp"
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
    void setPaletteRange(float minimum, float maximum);
    void setPalette(int palette);
    uint32_t getPalette() const;
    float getMinTemperature() const;
    float getMaxTemperature() const;
    bool isPaletteVisible() const { return paletteVisible; }
    void setSelectedProbe(uint32_t probeId, int probeIndex);
    void updateProbePosition(uint32_t probeId, int probeIndex, const glm::vec3& newPos);
    void updateProbeTemperatures(uint64_t nodeId, const std::vector<float>& temps);
    void setTextRenderer(ScreenTextRenderer* renderer);
    void renderSurface(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent);
    void renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent, const glm::mat4& viewProj);
    const std::vector<ProbeMarkerRenderer::RenderBinding>& getProbeBindings() const { return probeBindings; }
    void cleanup();

private:
    void rebuildBindings();

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    MemoryAllocator& memoryAllocator;
    std::unique_ptr<HeatSurfaceRenderer> surfaceRenderer;
    std::unique_ptr<VectorArrowRenderer> vectorArrowRenderer;
    std::unique_ptr<HeatBoxRenderer> boxRenderer;
    std::unique_ptr<ProbeMarkerRenderer> probeRenderer;
    std::unordered_map<uint64_t, HeatDisplayController::Config> configsBySocket;
    std::vector<HeatSurfaceRenderer::SurfaceRenderBinding> surfaceBindings;
    std::vector<VectorArrowRenderer::VectorRenderBinding> fluxVectorBindings;
    std::vector<HeatBoxRenderer::RenderBinding> boxBindings;
    std::vector<ProbeMarkerRenderer::RenderBinding> probeBindings;
    uint32_t maxFramesInFlight = 0;
    bool surfaceInitialized = false;
    bool overlayInitialized = false;
    bool paletteVisible = false;
    uint32_t selectedProbeId = 0;
    int selectedProbeIndex = -1;
};

}
