#pragma once

#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;

class HeatSystemPlayback {
public:
    struct Controls {
        bool paused = false;
        uint32_t resetCounter = 0;
        uint32_t rewindFrame = 0;
    };

    ~HeatSystemPlayback();

    bool initialize(VulkanDevice& device, MemoryAllocator& allocator, uint32_t nodeCount, uint32_t maxFrames);
    void cleanup();

    void recordFrame(VkCommandBuffer cmd, VkBuffer srcTempBuffer, VkDeviceSize srcOffset);

    void reset();

    uint32_t getRecordedFrames() const { return recordedFrames; }
    uint32_t getMaxFrames() const { return maxFrames; }
    uint32_t getNodeCount() const { return nodeCount; }
    bool isComplete() const { return recordedFrames >= maxFrames; }
    bool isValid() const { return historyBuffer != VK_NULL_HANDLE; }

    VkBuffer getHistoryBuffer() const { return historyBuffer; }
    VkDeviceSize getHistoryBufferOffset() const { return historyBufferOffset; }

private:
    MemoryAllocator* allocator = nullptr;
    VkBuffer historyBuffer = VK_NULL_HANDLE;
    VkDeviceSize historyBufferOffset = 0;
    uint32_t nodeCount = 0;
    uint32_t maxFrames = 0;
    uint32_t recordedFrames = 0;
    VkDeviceSize frameStride = 0;
};
