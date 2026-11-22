#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class CFDStreamlineStage {
public:
    static constexpr uint32_t MaxSegmentsPerLine = 256;

    CFDStreamlineStage(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool,
        uint32_t maxFramesInFlight);
    ~CFDStreamlineStage();

    bool initialize(
        const glm::vec3& domainMin,
        const glm::vec3& domainMax,
        const glm::vec3& volumeGridMin,
        const glm::ivec3& volumeGridDim,
        float volumeGridCellSize,
        const std::vector<glm::vec4>& seeds,
        VkBuffer seedToFluidNodeBuffer,
        VkDeviceSize seedToFluidNodeBufferOffset,
        uint32_t seedToFluidNodeCount,
        VkBuffer centroidBuffer,
        VkDeviceSize centroidBufferOffset);
    void cleanup();

    void recordCompute(
        VkCommandBuffer cmd,
        uint32_t currentFrame,
        VkBuffer fluidSeedGridBuffer,
        VkDeviceSize fluidSeedGridOffset,
        VkBuffer velocityBuffer,
        VkDeviceSize velocityBufferOffset,
        VkBuffer gradientsBuffer,
        VkDeviceSize gradientsBufferOffset,
        uint32_t fluidNodeCount,
        float stepLength);

    VkBuffer vertexBuffer() const { return outputBuffer; }
    VkDeviceSize vertexBufferOffset() const { return outputOffset; }
    uint32_t vertexCount() const { return seedCount * MaxSegmentsPerLine * 2; }
    bool isValid() const {
        return ready &&
            outputBuffer != VK_NULL_HANDLE &&
            seedToFluidNodeBuffer != VK_NULL_HANDLE &&
            centroidsBuffer != VK_NULL_HANDLE;
    }

private:
    bool createPipeline();
    bool createOutputBuffer(uint32_t seeds);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 0;

    glm::vec3 gridMin{0.0f};
    glm::ivec3 gridDim{0};
    float cellSize = 0.0f;
    glm::vec3 domainMin{0.0f};
    glm::vec3 domainMax{0.0f};

    VkBuffer seedBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedBufferOffset = 0;
    VkBuffer seedToFluidNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedToFluidNodeBufferOffset = 0;
    uint32_t seedToFluidNodeCount = 0;
    uint32_t seedCount = 0;

    VkBuffer centroidsBuffer = VK_NULL_HANDLE;
    VkDeviceSize centroidsBufferOffset = 0;

    VkBuffer outputBuffer = VK_NULL_HANDLE;
    VkDeviceSize outputOffset = 0;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool ready = false;
};
