#include <algorithm>
#include <iostream>
#include "MemoryPool.hpp"
#include "MemoryAllocator.hpp"
#include "VulkanDevice.hpp"

MemoryPool::MemoryPool(VulkanDevice& vulkanDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps)
    : vulkanDevice(vulkanDevice), memProperties(memProps), buffer(VK_NULL_HANDLE), memory(VK_NULL_HANDLE) {

    // Round pool size to power of 2
    poolSize = MemoryAllocator::roundUpPowerOfTwo(size);
    maxOrder = MemoryAllocator::ceilLog2(poolSize);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = poolSize;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateBuffer(vulkanDevice.getDevice(), &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        std::cerr << "[MemoryPool] Failed to create buffer" << std::endl;
        buffer = VK_NULL_HANDLE;
        return;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(vulkanDevice.getDevice(), buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = vulkanDevice.findMemoryType(memReqs.memoryTypeBits, memProps);

    result = vkAllocateMemory(vulkanDevice.getDevice(), &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        std::cerr << "[MemoryPool] Failed to allocate memory" << std::endl;
        vkDestroyBuffer(vulkanDevice.getDevice(), buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        return;
    }

    vkBindBufferMemory(vulkanDevice.getDevice(), buffer, memory, 0);

    if (memProps & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vkMapMemory(vulkanDevice.getDevice(), memory, 0, poolSize, 0, &mappedPtr);
    }

    // Initialize buddy free lists: one big free block at maxOrder
    freeList.resize(maxOrder + 1);
    freeList[maxOrder].push_back(0);
}

MemoryPool::~MemoryPool() {
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vulkanDevice.getDevice(), buffer, nullptr);
        buffer = VK_NULL_HANDLE;
    }

    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(vulkanDevice.getDevice(), memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
}

std::pair<VkBuffer, VkDeviceSize> MemoryPool::allocate(VkDeviceSize size) {
    uint32_t targetOrder = MemoryAllocator::ceilLog2(size);
    if (targetOrder > maxOrder) return {VK_NULL_HANDLE, 0};

    // Find smallest free block >= target order
    uint32_t order = targetOrder;
    while (order <= maxOrder && freeList[order].empty()) {
        order++;
    }
    if (order > maxOrder) return {VK_NULL_HANDLE, 0};

    // Split down to target order
    while (order > targetOrder) {
        VkDeviceSize blockOffset = freeList[order].back();
        freeList[order].pop_back();

        // Split into two buddies
        VkDeviceSize halfSize = static_cast<VkDeviceSize>(1) << (order - 1);
        VkDeviceSize buddyOffset = blockOffset + halfSize;
        freeList[order - 1].push_back(buddyOffset);
        freeList[order - 1].push_back(blockOffset);
        order--;
    }

    // Allocate from target order
    VkDeviceSize offset = freeList[targetOrder].back();
    freeList[targetOrder].pop_back();
    return {buffer, offset};
}

void MemoryPool::free(VkDeviceSize offset, uint32_t order) {
    // Coalesce with buddy
    while (order < maxOrder) {
        VkDeviceSize buddyOffset = offset ^ (static_cast<VkDeviceSize>(1) << order);

        // Check if buddy is free at this order
        auto& buddyList = freeList[order];
        auto it = std::find(buddyList.begin(), buddyList.end(), buddyOffset);
        if (it == buddyList.end()) break;

        // Remove buddy and coalesce
        std::swap(*it, buddyList.back());
        buddyList.pop_back();
        offset = std::min(offset, buddyOffset);
        order++;
    }

    freeList[order].push_back(offset);
}
