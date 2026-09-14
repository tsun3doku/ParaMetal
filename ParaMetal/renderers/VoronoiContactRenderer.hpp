#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>

#include "voronoi/VoronoiGpuStructs.hpp"

class VulkanDevice;
class UniformBufferManager;

class VoronoiContactRenderer {
public:
    VoronoiContactRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager);
    ~VoronoiContactRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void render(
        VkCommandBuffer cmdBuffer,
        uint32_t frameIndex,
        VkBuffer vertexBuffer,
        VkDeviceSize vertexBufferOffset,
        uint32_t vertexCount);

    void cleanup();

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);

    VulkanDevice& vulkanDevice;
    UniformBufferManager& uniformBufferManager;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
