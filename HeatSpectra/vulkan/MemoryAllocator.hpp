#pragma once
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include "util/Structs.hpp"
#include "MemoryPool.hpp"

class VulkanDevice;

class MemoryAllocator {
public:
    explicit MemoryAllocator(VulkanDevice& vulkanDevice) : vulkanDevice(vulkanDevice) {}
    ~MemoryAllocator();

    std::pair<VkBuffer, VkDeviceSize> allocate(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkDeviceSize alignment = 1);
    VkDeviceMemory allocateImageMemory(VkImage image, VkMemoryPropertyFlags memProps);

    void free(VkBuffer buffer, VkDeviceSize offset);
    void freeImageMemory(VkDeviceMemory memory);

    AllocatorStats getStats();
    void* getMappedPointer(VkBuffer buffer, VkDeviceSize offset);

    static VkDeviceSize computeBufferAlignment(const VulkanDevice& vulkanDevice, VkBufferUsageFlags usage);
    static uint32_t bitWidth(VkDeviceSize value);
    static bool isPowerOfTwo(VkDeviceSize value);
    static VkDeviceSize roundUpPowerOfTwo(VkDeviceSize value);
    static uint32_t ceilLog2(VkDeviceSize value);

private:
    struct AllocationInfo {
        VkBuffer buffer;
        VkDeviceSize offset;
        uint32_t order;
    };

    VulkanDevice& vulkanDevice;
    std::map<std::pair<VkBufferUsageFlags, VkMemoryPropertyFlags>, std::vector<std::unique_ptr<MemoryPool>>> pools;
    std::unordered_set<VkDeviceMemory> imageMemories;
    std::unordered_map<uint64_t, AllocationInfo> activeAllocations;

    const VkDeviceSize DEFAULT_POOL_SIZE = 256 * 1024 * 1024;
    std::mutex allocationMutex;

    MemoryPool& createNewPool(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps);

    static uint64_t makeAllocationKey(VkBuffer buffer, VkDeviceSize offset);
};
