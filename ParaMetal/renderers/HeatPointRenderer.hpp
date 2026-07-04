#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class VulkanDevice;
class UniformBufferManager;

class HeatPointRenderer {
public:
    struct PointRenderBinding {
        uint64_t domainKey = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        VkBuffer tempBuffer = VK_NULL_HANDLE;
        VkDeviceSize tempBufferOffset = 0;
        uint32_t pointCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        float pointSize = 0.01f;
    };

    HeatPointRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager);
    ~HeatPointRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void cleanup();
    void render(VkCommandBuffer commandBuffer, uint32_t frameIndex, const std::vector<PointRenderBinding>& bindings, VkExtent2D extent);

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);
    
    VkDescriptorSet allocateDescriptorSet(VkDescriptorPool pool);
    void updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex, const PointRenderBinding& binding);

    VulkanDevice& vulkanDevice;
    UniformBufferManager& uniformBufferManager;

    std::vector<VkDescriptorPool> descriptorPools;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
