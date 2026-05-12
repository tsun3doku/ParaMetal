#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class VulkanDevice;
class UniformBufferManager;

class HeatSurfaceRenderer {
public:
    struct SurfaceRenderBinding {
        uint32_t runtimeModelId = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexBufferOffset = 0;
        uint32_t indexCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        std::array<VkBufferView, 11> bufferViews{};
        VkBuffer surfaceBuffer = VK_NULL_HANDLE;
        VkDeviceSize surfaceBufferOffset = 0;
    };

    HeatSurfaceRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager);
    ~HeatSurfaceRenderer();

    void initialize(VkRenderPass renderPass, uint32_t maxFramesInFlight);
    void cleanup();
    void updateDescriptors(const std::vector<SurfaceRenderBinding>& surfaces, uint32_t maxFramesInFlight, bool forceReallocate);
    void render(VkCommandBuffer commandBuffer, uint32_t frameIndex, const std::vector<SurfaceRenderBinding>& surfaces) const;

private:
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSetLayout();
    bool createPipeline(VkRenderPass renderPass);
    void drawModel(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, const SurfaceRenderBinding& binding) const;
    bool updateDescriptorSetVector(
        const std::array<VkBufferView, 11>& bufferViews,
        VkBuffer surfaceBuffer,
        VkDeviceSize surfaceBufferOffset,
        uint32_t maxFramesInFlight,
        std::vector<VkDescriptorSet>& targetSets,
        bool forceReallocate);

    VulkanDevice& vulkanDevice;
    UniformBufferManager& uniformBufferManager;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    std::unordered_map<uint32_t, std::vector<VkDescriptorSet>> surfaceDescriptorSets;

    bool initialized = false;
};
