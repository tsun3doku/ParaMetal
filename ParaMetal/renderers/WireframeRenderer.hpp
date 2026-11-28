#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <cstdint>

class VulkanDevice;
class UniformBufferManager;

class WireframeRenderer {
public:
    struct PushConstant {
        alignas(16) glm::mat4 modelMatrix{1.0f};
        alignas(16) glm::vec3 color{0.05f};
        float padding = 0.0f;
    };

    WireframeRenderer(VulkanDevice& device, VkDescriptorSetLayout geometryDescriptorSetLayout,
                     VkRenderPass renderPass, uint32_t subpass);
    ~WireframeRenderer();
    
    void bindPipeline(VkCommandBuffer cmdBuffer);
    void renderModel(
        VkCommandBuffer cmdBuffer,
        VkDescriptorSet geometryDescriptorSet,
        const glm::mat4& modelMatrix,
        VkBuffer vertexBuffer,
        VkDeviceSize vertexBufferOffset,
        VkBuffer indexBuffer,
        VkDeviceSize indexBufferOffset,
        uint32_t indexCount,
        const glm::vec3& color);
    
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }
    
    bool recreatePipeline(VkRenderPass renderPass, uint32_t subpass);
    void cleanup();

private:
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);

    VulkanDevice& vulkanDevice;
    VkDescriptorSetLayout geometryDescriptorSetLayout;
    
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    
    bool initialized = false;
};
