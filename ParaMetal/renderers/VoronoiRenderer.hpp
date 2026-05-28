#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

class VulkanDevice;
class UniformBufferManager;
class CommandPool;
class MemoryAllocator;

class VoronoiRenderer {
public:
    struct VoronoiRenderBinding {
        uint64_t bindingKey = 0;
        uint32_t runtimeModelId = 0;
        uint32_t vertexCount = 0;
        VkBuffer seedBuffer = VK_NULL_HANDLE;
        VkDeviceSize seedOffset = 0;
        VkBuffer neighborBuffer = VK_NULL_HANDLE;
        VkDeviceSize neighborOffset = 0;
        VkBufferView supportingHalfedgeView = VK_NULL_HANDLE;
        VkBufferView supportingAngleView = VK_NULL_HANDLE;
        VkBufferView halfedgeView = VK_NULL_HANDLE;
        VkBufferView edgeView = VK_NULL_HANDLE;
        VkBufferView triangleView = VK_NULL_HANDLE;
        VkBufferView lengthView = VK_NULL_HANDLE;
        VkBufferView inputHalfedgeView = VK_NULL_HANDLE;
        VkBufferView inputEdgeView = VK_NULL_HANDLE;
        VkBufferView inputTriangleView = VK_NULL_HANDLE;
        VkBufferView inputLengthView = VK_NULL_HANDLE;
        VkBuffer candidateBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateOffset = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexOffset = 0;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexOffset = 0;
        uint32_t indexCount = 0;
        glm::mat4 modelMatrix{1.0f};
    };

    VoronoiRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& commandPool);
    ~VoronoiRenderer();

    void initialize(VkRenderPass renderPass, uint32_t maxFramesInFlight);
    
    void updateDescriptors(const std::vector<VoronoiRenderBinding>& bindings, uint32_t maxFramesInFlight, bool forceReallocate);
    void render(VkCommandBuffer cmd, uint32_t frameIndex, const std::vector<VoronoiRenderBinding>& bindings) const;

    void cleanup();

private:
    uint32_t calculateMipLevels(uint32_t width, uint32_t height);
    bool createWireframeTexture();

    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass);
    bool updateDescriptorSetForBinding(const VoronoiRenderBinding& binding, uint32_t maxFramesInFlight, std::vector<VkDescriptorSet>& targetSets, bool forceReallocate);
    void drawBinding(VkCommandBuffer cmd, VkDescriptorSet descriptorSet, const VoronoiRenderBinding& binding) const;
    VulkanDevice& vulkanDevice;
    MemoryAllocator& allocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& renderCommandPool;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> voronoiDescriptorSets;

    VkImage wireframeTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory wireframeTextureMemory = VK_NULL_HANDLE;
    VkImageView wireframeTextureView = VK_NULL_HANDLE;
    VkSampler wireframeTextureSampler = VK_NULL_HANDLE;
    
    bool initialized = false;
    
};
