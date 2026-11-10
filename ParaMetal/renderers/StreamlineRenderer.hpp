#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

class VulkanDevice;
class MemoryAllocator;
class UniformBufferManager;
class CommandPool;

namespace render {

// Draws the line vertex buffer integrated by CFDStreamlineStage. One renderer
// instance serves every displayed CFD-driven streamline set (CFD node or Ember).
class StreamlineRenderer {
public:
    struct RenderBinding {
        uint64_t bindingKey = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        uint32_t vertexCount = 0;
        uint32_t palette = 0;
        float minSpeed = 0.0f;
        float maxSpeed = 1.0f;
    };

    struct PushConstants {
        float minSpeed = 0.0f;
        float maxSpeed = 1.0f;
        uint32_t palette = 0;
        float padding = 0.0f;
    };

    StreamlineRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~StreamlineRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void render(VkCommandBuffer commandBuffer, uint32_t frameIndex, const std::vector<RenderBinding>& lines);
    void cleanup();

private:
    bool createPaletteTextures();
    void destroyPaletteTextures();
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);

    VkDescriptorSet allocateDescriptorSet(VkDescriptorPool pool);
    void updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descriptorPools;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    VkImage infernoImage = VK_NULL_HANDLE;
    VkDeviceMemory infernoMemory = VK_NULL_HANDLE;
    VkImageView infernoView = VK_NULL_HANDLE;
    VkImage viridisImage = VK_NULL_HANDLE;
    VkDeviceMemory viridisMemory = VK_NULL_HANDLE;
    VkImageView viridisView = VK_NULL_HANDLE;
    VkImage inferno2Image = VK_NULL_HANDLE;
    VkDeviceMemory inferno2Memory = VK_NULL_HANDLE;
    VkImageView inferno2View = VK_NULL_HANDLE;
    VkImage parulaImage = VK_NULL_HANDLE;
    VkDeviceMemory parulaMemory = VK_NULL_HANDLE;
    VkImageView parulaView = VK_NULL_HANDLE;
    VkSampler paletteSampler = VK_NULL_HANDLE;

    bool initialized = false;
};

} // namespace render
