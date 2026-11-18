#pragma once

#include <array>
#include <cstddef>
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
        bool outline = true;
    };

    struct Highlight {
        int selectedFace = -1;
        int hoverFace = -1;
        std::array<glm::vec3, 8> corners{};
        bool boundsValid = false;
    };

    struct PushConstants {
        glm::vec4 boxMin{0.0f};
        glm::vec4 boxMax{0.0f};
        glm::vec4 selectedColor{1.0f, 1.0f, 1.0f, 0.14f};
        glm::vec4 hoverColor{1.0f, 0.75f, 0.35f, 0.10f};
        int selectedFace = -1;
        int hoverFace = -1;
    };

    static_assert(sizeof(PushConstants) == 72, "HeatBoxRenderer push constants must stay 72 bytes");
    static_assert(offsetof(PushConstants, selectedFace) == 64, "HeatBoxRenderer selected face offset changed");
    static_assert(offsetof(PushConstants, hoverFace) == 68, "HeatBoxRenderer hover face offset changed");

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
        const std::vector<RenderBinding>& bindings,
        const Highlight& highlight);

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
    VkPipeline facePipeline = VK_NULL_HANDLE;

    bool initialized = false;
};
