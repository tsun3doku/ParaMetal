#pragma once

#include "runtime/PointDisplayController.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>

class PointRenderer;
class UniformBufferManager;
class VulkanDevice;

namespace render {

class PointOverlayRenderer {
public:
    PointOverlayRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager);
    ~PointOverlayRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void apply(uint64_t socketKey, const PointDisplayController::Config& config);
    void remove(uint64_t socketKey);
    void render(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent);
    void cleanup();

private:
    VulkanDevice& vulkanDevice;
    UniformBufferManager& uniformBufferManager;

    std::unique_ptr<PointRenderer> pointRenderer;
    std::unordered_map<uint64_t, PointDisplayController::Config> configsBySocket;
    uint32_t maxFramesInFlight = 0;
    bool initialized = false;
};

} // namespace render
