#pragma once

#include "runtime/RemeshDisplayController.hpp"

#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>

class VulkanDevice;
class MemoryAllocator;
class UniformBufferManager;
class CommandPool;
class iODT;

class IntrinsicRenderer {
public:
    IntrinsicRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& commandPool);
    ~IntrinsicRenderer();
    void cleanup();

    void apply(uint64_t socketKey, const RemeshDisplayController::Config& config);
    void remove(uint64_t socketKey);
    bool initializeSurface(VkRenderPass renderPass, uint32_t maxFramesInFlight, uint32_t subpassIndex);
    bool initializeOverlay(VkRenderPass renderPass, uint32_t maxFramesInFlight, uint32_t subpassIndex);
    void renderSurface(VkCommandBuffer commandBuffer, uint32_t currentFrame);
    void renderOverlay(VkCommandBuffer commandBuffer, uint32_t currentFrame);

private:
    uint32_t calculateMipLevels(uint32_t width, uint32_t height);
    bool createWireframeTexture();
    void pruneStaleSocketResources();
    void releaseDescriptorSetsForSocket(uint64_t socketKey);
    void allocateSupportingHalfedgeDescriptorSets(uint64_t socketKey, uint32_t maxFramesInFlight);
    void allocateFaceNormalDescriptorSets(uint64_t socketKey, uint32_t maxFramesInFlight);
    void allocateVertexNormalDescriptorSets(uint64_t socketKey, uint32_t maxFramesInFlight);
    void updateSupportingHalfedgeDescriptorSet(uint64_t socketKey, const RemeshDisplayController::Config& config, uint32_t currentFrame);
    void updateFaceNormalDescriptorSet(uint64_t socketKey, const RemeshDisplayController::Config& config, uint32_t currentFrame);
    void updateVertexNormalDescriptorSet(uint64_t socketKey, const RemeshDisplayController::Config& config, uint32_t currentFrame);

    bool createSupportingHalfedgeDescriptorPool(uint32_t maxFramesInFlight);
    bool createSupportingHalfedgeDescriptorSetLayout();

    bool createIntrinsicNormalsDescriptorPool(uint32_t maxFramesInFlight);
    bool createIntrinsicNormalsDescriptorSetLayout();
    bool createIntrinsicVertexNormalsDescriptorPool(uint32_t maxFramesInFlight);
    bool createIntrinsicVertexNormalsDescriptorSetLayout();

    bool createSupportingHalfedgePipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    bool createIntrinsicNormalsPipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    bool createIntrinsicVertexNormalsPipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    void renderIntrinsicNormals(VkCommandBuffer commandBuffer, uint32_t currentFrame);
    void renderIntrinsicVertexNormals(VkCommandBuffer commandBuffer, uint32_t currentFrame);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& allocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& renderCommandPool;
    uint32_t maxFramesInFlight = 0;

    VkDescriptorPool supportingHalfedgeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout supportingHalfedgeDescriptorSetLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> supportingHalfedgeDescriptorSetsBySocket;

    VkImage wireframeTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory wireframeTextureMemory = VK_NULL_HANDLE;
    VkImageView wireframeTextureView = VK_NULL_HANDLE;
    VkSampler wireframeTextureSampler = VK_NULL_HANDLE;

    VkDescriptorPool intrinsicNormalsDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout intrinsicNormalsDescriptorSetLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> intrinsicNormalsDescriptorSetsBySocket;

    VkDescriptorPool intrinsicVertexNormalsDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout intrinsicVertexNormalsDescriptorSetLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> intrinsicVertexNormalsDescriptorSetsBySocket;
    std::unordered_map<uint64_t, RemeshDisplayController::Config> remeshConfigsBySocketKey;

    VkPipeline supportingHalfedgePipeline = VK_NULL_HANDLE;
    VkPipelineLayout supportingHalfedgePipelineLayout = VK_NULL_HANDLE;
    VkPipeline intrinsicNormalsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout intrinsicNormalsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline intrinsicVertexNormalsPipeline = VK_NULL_HANDLE;
    VkPipelineLayout intrinsicVertexNormalsPipelineLayout = VK_NULL_HANDLE;

    bool surfaceInitialized = false;
    bool overlayInitialized = false;
};
