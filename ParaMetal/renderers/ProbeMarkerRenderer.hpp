#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

class CommandPool;
class MemoryAllocator;
class UniformBufferManager;
class VulkanDevice;
class ScreenTextRenderer;

class ProbeMarkerRenderer {
public:
    struct RenderBinding {
        glm::vec3 worldPosition{0.0f};
        glm::vec4 color{1.0f};
        float scale = 1.0f;
        bool isSelected = false;
        uint32_t probeId = 0;
        uint64_t nodeId = 0;
        std::string probeName;
        float currentTemperatureC = 20.0f;
        bool showTemperature = true;
        bool showName = true;
    };

    ProbeMarkerRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~ProbeMarkerRenderer();

    void setTextRenderer(ScreenTextRenderer* renderer) { screenTextRenderer = renderer; }

    void initialize(
        VkRenderPass renderPass,
        uint32_t subpass,
        uint32_t maxFramesInFlight);

    void render(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        VkExtent2D extent,
        const glm::mat4& viewProj,
        const std::vector<RenderBinding>& bindings);

    void cleanup();

private:
    struct PushConstants {
        glm::vec4 worldPos;      // xyz = worldPosition, w = scale
        glm::vec4 colorTint;     // rgba tint
        glm::vec2 viewportSize;  // width, height in pixels
        glm::vec2 markerSize;    // width, height in pixels (e.g. 32.0, 36.0)
        uint32_t isSelected = 0; // 0 = unselected, 1 = selected
        uint32_t pad0 = 0;
        uint32_t pad1 = 0;
        uint32_t pad2 = 0;
    };

    bool createTexture();
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);
    void renderLabels(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        VkExtent2D extent,
        const glm::mat4& viewProj,
        const std::vector<RenderBinding>& bindings);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;
    ScreenTextRenderer* screenTextRenderer = nullptr;

    VkImage textureImage = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory = VK_NULL_HANDLE;
    VkImageView textureImageView = VK_NULL_HANDLE;
    VkSampler textureSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
