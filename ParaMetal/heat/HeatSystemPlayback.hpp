#pragma once

#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;

class HeatSystemPlayback {
public:
    ~HeatSystemPlayback();

    bool initialize(VulkanDevice& device, MemoryAllocator& allocator, uint32_t nodeCount, uint32_t frameCapacity);
    void cleanup();

    void recordFrame(VkCommandBuffer cmd, VkBuffer srcTempBuffer, VkDeviceSize srcOffset);

    void reset();

    uint32_t getRecordedFrameCount() const { return recordedFrames; }
    uint32_t getFrameCapacity() const { return frameCapacity; }
    uint32_t getNodeCount() const { return nodeCount; }
    VkDeviceSize getFrameStride() const { return frameStride; }
    bool isValid() const { return historyBuffer != VK_NULL_HANDLE; }

    VkBuffer getHistoryBuffer() const { return historyBuffer; }
    VkDeviceSize getHistoryBufferOffset() const { return historyBufferOffset; }

private:
    MemoryAllocator* allocator = nullptr;
    VkBuffer historyBuffer = VK_NULL_HANDLE;
    VkDeviceSize historyBufferOffset = 0;
    uint32_t nodeCount = 0;
    uint32_t frameCapacity = 0;
    uint32_t recordedFrames = 0;
    VkDeviceSize frameStride = 0;
};
