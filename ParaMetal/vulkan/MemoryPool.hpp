#pragma once
#include <vector>
#include <vulkan/vulkan.h>

class VulkanDevice;

class MemoryPool {
public:
    MemoryPool(VulkanDevice& vulkanDevice, VkDeviceSize poolSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps);
    ~MemoryPool();

    std::pair<VkBuffer, VkDeviceSize> allocate(VkDeviceSize size);
    void free(VkDeviceSize offset, uint32_t order);

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkMemoryPropertyFlags memProperties = 0;
    void* mappedPtr = nullptr;

    VkDeviceSize getPoolSize() const { return poolSize; }
    uint32_t getMaxOrder() const { return maxOrder; }
    const std::vector<std::vector<VkDeviceSize>>& getFreeLists() const { return freeList; }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

private:
    VulkanDevice& vulkanDevice;
    VkDeviceSize poolSize = 0;
    uint32_t maxOrder = 0;
    std::vector<std::vector<VkDeviceSize>> freeList;
};
