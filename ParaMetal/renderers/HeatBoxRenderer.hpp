#pragma once

#include <array>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

class MemoryAllocator;
class UniformBufferManager;
class VulkanDevice;

class HeatBoxRenderer {
public:
    struct RenderBinding {
        std::array<glm::vec3, 8> corners{};
    };

    HeatBoxRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager);
    ~HeatBoxRenderer();

    void initialize(
        VkRenderPass renderPass,
        uint32_t subpass,
        uint32_t maxFramesInFlight);

    void render(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const std::vector<RenderBinding>& bindings);

    void cleanup();

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
