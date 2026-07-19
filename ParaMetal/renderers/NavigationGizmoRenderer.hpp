#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>

class CommandPool;
class MemoryAllocator;
class VulkanDevice;
struct NavigationGizmoRenderData;

struct NavigationGizmoVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec2 uv{};

    static VkVertexInputBindingDescription bindingDescription();
    static std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions();
};

class NavigationGizmoRenderer {
public:
    NavigationGizmoRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        CommandPool& commandPool,
        VkRenderPass renderPass,
        uint32_t subpassIndex);
    ~NavigationGizmoRenderer();

    bool isReady() const { return ready; }
    void render(VkCommandBuffer commandBuffer, const NavigationGizmoRenderData& data);
    void cleanup();

private:
    struct PushConstants {
        glm::mat4 rotation{1.0f};
        uint32_t hoveredRegion = 0;
        uint32_t pressedRegion = 0;
    };

    bool createGeometry();
    bool createLabelTexture();
    bool createDescriptors();
    bool createPipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& memory);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkImage labelAtlasImage = VK_NULL_HANDLE;
    VkDeviceMemory labelAtlasMemory = VK_NULL_HANDLE;
    VkImageView labelAtlasView = VK_NULL_HANDLE;
    VkSampler labelAtlasSampler = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexMemory = VK_NULL_HANDLE;
    std::array<glm::vec3, 6> faceNormals{};
    bool ready = false;
};
