#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

class MemoryAllocator;
class VulkanDevice;

namespace heat {
struct TimeUniform;
}

class HeatSystemSimRuntime {
public:
    bool initialize(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator);
    void reset();
    void cleanup(MemoryAllocator& memoryAllocator);

    bool isInitialized() const { return timeBuffer != VK_NULL_HANDLE; }

    VkBuffer getTimeBuffer() const { return timeBuffer; }
    VkDeviceSize getTimeBufferOffset() const { return timeBufferOffset; }
    heat::TimeUniform* getMappedTimeData() const;

private:
    VkBuffer timeBuffer = VK_NULL_HANDLE;
    VkDeviceSize timeBufferOffset = 0;
    void* mappedTimeData = nullptr;
};
