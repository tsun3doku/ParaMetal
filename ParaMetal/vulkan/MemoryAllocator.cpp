#include <algorithm>
#include <iostream>
#include "VulkanDevice.hpp"
#include "MemoryAllocator.hpp"

VkDeviceSize MemoryAllocator::computeBufferAlignment(const VulkanDevice& vulkanDevice, VkBufferUsageFlags usage) {
    VkDeviceSize requiredAlignment = 1;

    const VkPhysicalDeviceLimits& limits = vulkanDevice.getPhysicalDeviceProperties().limits;

    if ((usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0) {
        requiredAlignment = std::max(requiredAlignment, limits.minUniformBufferOffsetAlignment);
    }
    if ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        requiredAlignment = std::max(requiredAlignment, limits.minStorageBufferOffsetAlignment);
    }
    if ((usage & (VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT)) != 0) {
        requiredAlignment = std::max(requiredAlignment, limits.minTexelBufferOffsetAlignment);
    }

    return requiredAlignment;
}

uint32_t MemoryAllocator::bitWidth(VkDeviceSize value) {
    if (value == 0) return 0;
    uint32_t bits = 0;
    while (value > 0) {
        value >>= 1;
        bits++;
    }
    return bits;
}

bool MemoryAllocator::isPowerOfTwo(VkDeviceSize value) {
    return value > 0 && (value & (value - 1)) == 0;
}

VkDeviceSize MemoryAllocator::roundUpPowerOfTwo(VkDeviceSize value) {
    if (value == 0) return 1;
    if (isPowerOfTwo(value)) return value;
    return static_cast<VkDeviceSize>(1) << bitWidth(value);
}

uint32_t MemoryAllocator::ceilLog2(VkDeviceSize value) {
    if (value <= 1) return 0;
    if (isPowerOfTwo(value)) return bitWidth(value) - 1;
    return bitWidth(value);
}

uint64_t MemoryAllocator::makeAllocationKey(VkBuffer buffer, VkDeviceSize offset) {
    // Hash combine: pointer value + offset
    uint64_t h = reinterpret_cast<uint64_t>(buffer);
    h ^= (offset + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    return h;
}

MemoryAllocator::~MemoryAllocator() {
    std::lock_guard<std::mutex> lock(allocationMutex);

    for (VkDeviceMemory imageMemory : imageMemories) {
        if (imageMemory != VK_NULL_HANDLE) {
            vkFreeMemory(vulkanDevice.getDevice(), imageMemory, nullptr);
        }
    }
    imageMemories.clear();

    activeAllocations.clear();

    for (auto& pair : pools) {
        pair.second.clear();
    }
    pools.clear();
}

std::pair<VkBuffer, VkDeviceSize> MemoryAllocator::allocate(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps, VkDeviceSize alignment) {
    std::lock_guard<std::mutex> lock(allocationMutex);

    const VkDeviceSize requiredAlignment = computeBufferAlignment(vulkanDevice, usage);
    VkDeviceSize effectiveAlignment = std::max(size, requiredAlignment);
    VkDeviceSize alignedSize = roundUpPowerOfTwo(effectiveAlignment);
    uint32_t order = ceilLog2(alignedSize);

    auto key = std::make_pair(usage, memProps);

    // Try existing pools
    for (auto& poolPtr : pools[key]) {
        MemoryPool& pool = *poolPtr;
        auto [buffer, offset] = pool.allocate(alignedSize);
        if (buffer != VK_NULL_HANDLE) {
            activeAllocations[makeAllocationKey(buffer, offset)] = {buffer, offset, order};
            return {buffer, offset};
        }
    }

    // Create new pool
    VkDeviceSize poolSize = std::max(DEFAULT_POOL_SIZE, alignedSize);
    pools[key].push_back(std::make_unique<MemoryPool>(vulkanDevice, poolSize, usage, memProps));
    MemoryPool& newPool = *pools[key].back();

    auto [buffer, offset] = newPool.allocate(alignedSize);
    if (buffer != VK_NULL_HANDLE) {
        activeAllocations[makeAllocationKey(buffer, offset)] = {buffer, offset, order};
    }
    return {buffer, offset};
}

VkDeviceMemory MemoryAllocator::allocateImageMemory(VkImage image, VkMemoryPropertyFlags memProps) {
    if (image == VK_NULL_HANDLE) {
        std::cerr << "[MemoryAllocator] Cannot allocate memory for null image" << std::endl;
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements memReqs{};
    vkGetImageMemoryRequirements(vulkanDevice.getDevice(), image, &memReqs);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = vulkanDevice.findMemoryType(memReqs.memoryTypeBits, memProps);

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(vulkanDevice.getDevice(), &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        std::cerr << "[MemoryAllocator] Failed to allocate image memory" << std::endl;
        return VK_NULL_HANDLE;
    }

    if (vkBindImageMemory(vulkanDevice.getDevice(), image, memory, 0) != VK_SUCCESS) {
        std::cerr << "[MemoryAllocator] Failed to bind image memory" << std::endl;
        vkFreeMemory(vulkanDevice.getDevice(), memory, nullptr);
        return VK_NULL_HANDLE;
    }

    {
        std::lock_guard<std::mutex> lock(allocationMutex);
        imageMemories.insert(memory);
    }

    return memory;
}

MemoryPool& MemoryAllocator::createNewPool(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps) {
    auto key = std::make_pair(usage, memProps);
    VkDeviceSize poolSize = std::max(DEFAULT_POOL_SIZE, roundUpPowerOfTwo(size));
    pools[key].push_back(std::make_unique<MemoryPool>(vulkanDevice, poolSize, usage, memProps));
    return *pools[key].back();
}

AllocatorStats MemoryAllocator::getStats() {
    AllocatorStats stats = {};
    for (auto& pair : pools) {
        auto& poolVector = pair.second;
        for (auto& poolPtr : poolVector) {
            MemoryPool& pool = *poolPtr;
            stats.totalAllocated += pool.getPoolSize();

            uint32_t maxOrder = pool.getMaxOrder();
            const auto& freeLists = pool.getFreeLists();

            // Count free space
            VkDeviceSize freeSpace = 0;
            for (uint32_t order = 0; order <= maxOrder; ++order) {
                if (order < freeLists.size()) {
                    freeSpace += freeLists[order].size() * (static_cast<VkDeviceSize>(1) << order);
                }
            }

            stats.usedBytes += pool.getPoolSize() - freeSpace;
        }
    }
    stats.allocationCount = static_cast<uint32_t>(activeAllocations.size());
    return stats;
}

void* MemoryAllocator::getMappedPointer(VkBuffer buffer, VkDeviceSize offset) {
    for (auto& poolPair : pools) {
        for (auto& poolPtr : poolPair.second) {
            MemoryPool& pool = *poolPtr;
            if (pool.buffer == buffer) {
                if (pool.mappedPtr) {
                    return static_cast<char*>(pool.mappedPtr) + offset;
                }
                return nullptr;
            }
        }
    }
    return nullptr;
}

void MemoryAllocator::free(VkBuffer buffer, VkDeviceSize offset) {
    std::lock_guard<std::mutex> lock(allocationMutex);

    uint64_t key = makeAllocationKey(buffer, offset);
    auto it = activeAllocations.find(key);
    if (it == activeAllocations.end()) return;

    AllocationInfo record = it->second;
    activeAllocations.erase(it);

    // Find pool and free
    for (auto& pair : pools) {
        auto& poolVector = pair.second;
        for (auto& poolPtr : poolVector) {
            MemoryPool& pool = *poolPtr;
            if (pool.buffer == record.buffer) {
                pool.free(record.offset, record.order);
                return;
            }
        }
    }
}

void MemoryAllocator::freeImageMemory(VkDeviceMemory memory) {
    if (memory == VK_NULL_HANDLE) {
        return;
    }

    std::lock_guard<std::mutex> lock(allocationMutex);
    const auto it = imageMemories.find(memory);
    if (it == imageMemories.end()) {
        return;
    }

    vkFreeMemory(vulkanDevice.getDevice(), memory, nullptr);
    imageMemories.erase(it);
}
