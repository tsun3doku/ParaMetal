#pragma once

#include "runtime/SubselectionDisplayController.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <utility>
#include <vector>

class MemoryAllocator;
class ModelRegistry;
class UniformBufferManager;
class VulkanDevice;
struct Vertex;

class SubselectionRenderer {
public:
    SubselectionRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager);
    ~SubselectionRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void apply(uint64_t socketKey, const SubselectionDisplayController::Config& config);
    void remove(uint64_t socketKey);
    void render(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        VkExtent2D extent,
        ModelRegistry& modelRegistry);
    void cleanup();

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);
    void buildLineVertices(const SubselectionDisplayController::Config& config, std::vector<Vertex>& outVertices) const;
    void buildPointVertices(const SubselectionDisplayController::Config& config, std::vector<Vertex>& outVertices) const;
    void uploadLineVertices(uint32_t frameIndex, const SubselectionDisplayController::Config& config);
    struct FrameBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize capacity = 0;
        uint64_t uploadedRevision = UINT64_MAX;
        uint32_t uploadedModelId = 0;
        VkBuffer lineBuffer = VK_NULL_HANDLE;
        VkDeviceSize lineOffset = 0;
        VkDeviceSize lineCapacity = 0;
        uint32_t lineVertexCount = 0;
        uint32_t pointFirstVertex = 0;
        uint32_t pointVertexCount = 0;
        uint64_t uploadedLineRevision = UINT64_MAX;
        uint32_t uploadedLineModelId = 0;
    };

    void rebuildSelection(
        const SubselectionDisplayController::Config& config,
        ModelRegistry& modelRegistry);
    void uploadFrameIndices(
        uint32_t frameIndex,
        const SubselectionDisplayController::Config& config);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline linePipeline = VK_NULL_HANDLE;
    VkPipeline pointPipeline = VK_NULL_HANDLE;

    std::vector<FrameBuffer> frameBuffers;
    std::vector<uint32_t> selectedIndices;
    std::vector<unsigned char> cachedLineVertexData;
    uint32_t cachedLineVertexCount = 0;
    uint32_t cachedPointFirstVertex = 0;
    uint32_t cachedPointVertexCount = 0;
    uint64_t cachedRevision = UINT64_MAX;
    uint32_t cachedModelId = 0;
    uint64_t cachedLineRevision = UINT64_MAX;
    uint32_t cachedLineModelId = 0;
    uint64_t activeSocketKey = 0;
    SubselectionDisplayController::Config config{};
    bool hasConfig = false;

    bool initialized = false;
};
