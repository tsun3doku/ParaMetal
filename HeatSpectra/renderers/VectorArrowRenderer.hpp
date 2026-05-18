#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>

class MemoryAllocator;
class UniformBufferManager;
class VulkanDevice;
class CommandPool;

class VectorArrowRenderer {
public:
    struct VectorRenderBinding {
        uint64_t bindingKey = 0;
        VkBuffer surfaceBuffer = VK_NULL_HANDLE;
        VkDeviceSize surfaceBufferOffset = 0;
        VkBuffer gradientBuffer = VK_NULL_HANDLE;
        VkDeviceSize gradientBufferOffset = 0;
        uint32_t sampleCount = 0;
        glm::mat4 modelMatrix{1.0f};
        float scale = 1.0f;
    };

    VectorArrowRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& commandPool);
    ~VectorArrowRenderer();

    void initialize(VkRenderPass renderPass, uint32_t maxFramesInFlight);
    void updateDescriptors(const std::vector<VectorRenderBinding>& vectors, uint32_t maxFramesInFlight, bool forceReallocate);
    void render(VkCommandBuffer commandBuffer, uint32_t frameIndex, const std::vector<VectorRenderBinding>& vectors) const;
    void cleanup();

private:
    struct PushConstants {
        alignas(16) glm::mat4 modelMatrix{1.0f};
        float scale = 1.0f;
        float normalOffset = 0.001f;
        float minLength = 0.00025f;
        float baseLength = 0.005f;
    };

    bool createArrowGeometry();
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass);
    bool updateDescriptorSetVector(const VectorRenderBinding& vector, uint32_t maxFramesInFlight, std::vector<VkDescriptorSet>& targetSets);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize vertexBufferOffset = 0;
    uint32_t vertexCount = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> vectorDescriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
