#include "VulkanBuffer.hpp"
#include "CommandBufferManager.hpp"

#include <cstring>
#include <iostream>

VkResult createStorageBuffer(MemoryAllocator& allocator, VulkanDevice& device, const void* data, VkDeviceSize size, VkBuffer& outBuffer, VkDeviceSize& outOffset,
    void** outMappedPtr, bool hostVisible, VkBufferUsageFlags additionalUsage) {
    freeBuffer(allocator, outBuffer, outOffset);
    if (outMappedPtr) {
        *outMappedPtr = nullptr;
    }

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | additionalUsage;
    VkMemoryPropertyFlags properties = hostVisible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkDeviceSize alignment = device.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    auto [buffer, offset] = allocator.allocate(size, usage, properties, alignment);
    outBuffer = buffer;
    outOffset = offset;

    if (hostVisible) {
        void* mappedPtr = allocator.getMappedPointer(buffer, offset);
        if (!mappedPtr) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }

        if (data) {
            memcpy(mappedPtr, data, size);
        }

        if (outMappedPtr) {
            *outMappedPtr = mappedPtr;
        }
    }

    return VK_SUCCESS;
}

VkResult createTexelBuffer(MemoryAllocator& allocator, VulkanDevice& device, const void* data, VkDeviceSize size, VkFormat format, VkBuffer& outBuffer,
    VkDeviceSize& outOffset, VkBufferView& outBufferView, VkBufferUsageFlags additionalUsage, VkDeviceSize alignment) {
    freeBufferView(device, outBufferView);
    freeBuffer(allocator, outBuffer, outOffset);

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | additionalUsage;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    auto [buffer, offset] = allocator.allocate(size, usage, properties, alignment);
    outBuffer = buffer;
    outOffset = offset;

    if (data) {
        void* mapped = allocator.getMappedPointer(buffer, offset);
        if (!mapped) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        memcpy(mapped, data, size);
    }

    VkBufferViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
    viewInfo.buffer = buffer;
    viewInfo.format = format;
    viewInfo.offset = offset;
    viewInfo.range = size;

    const VkResult viewResult = vkCreateBufferView(device.getDevice(), &viewInfo, nullptr, &outBufferView);
    if (viewResult != VK_SUCCESS) {
        return viewResult;
    }

    return VK_SUCCESS;
}

VkResult createStagingBuffer(MemoryAllocator& allocator, VkDeviceSize size, VkBuffer& outBuffer, VkDeviceSize& outOffset, void** outMappedPtr) {
    freeBuffer(allocator, outBuffer, outOffset);
    if (outMappedPtr) {
        *outMappedPtr = nullptr;
    }

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    auto [buffer, offset] = allocator.allocate(size, usage, properties);
    outBuffer = buffer;
    outOffset = offset;

    void* mappedPtr = allocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    if (outMappedPtr) {
        *outMappedPtr = mappedPtr;
    }

    return VK_SUCCESS;
}

VkResult createDownloadStagingBuffer(MemoryAllocator& allocator, VkDeviceSize size, VkBuffer& outBuffer, VkDeviceSize& outOffset, void** outMappedPtr) {
    freeBuffer(allocator, outBuffer, outOffset);
    if (outMappedPtr) {
        *outMappedPtr = nullptr;
    }

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    auto [buffer, offset] = allocator.allocate(size, usage, properties);
    outBuffer = buffer;
    outOffset = offset;

    void* mappedPtr = allocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    if (outMappedPtr) {
        *outMappedPtr = mappedPtr;
    }

    return VK_SUCCESS;
}

VkResult createUniformBuffer(MemoryAllocator& allocator, VulkanDevice& device, VkDeviceSize size, VkBuffer& outBuffer, VkDeviceSize& outOffset, void** outMappedPtr) {
    freeBuffer(allocator, outBuffer, outOffset);
    if (outMappedPtr) {
        *outMappedPtr = nullptr;
    }

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkDeviceSize alignment = device.getPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment;

    auto [buffer, offset] = allocator.allocate(size, usage, properties, alignment);
    outBuffer = buffer;
    outOffset = offset;

    void* mappedPtr = allocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    if (outMappedPtr) {
        *outMappedPtr = mappedPtr;
    }

    return VK_SUCCESS;
}

VkResult createVertexBuffer(MemoryAllocator& allocator, VkDeviceSize size, VkBuffer& outBuffer, VkDeviceSize& outOffset, VkBufferUsageFlags additionalUsage) {
    freeBuffer(allocator, outBuffer, outOffset);

    if (size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | additionalUsage;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkDeviceSize alignment = 16;

    auto [buffer, offset] = allocator.allocate(size, usage, properties, alignment);
    outBuffer = buffer;
    outOffset = offset;
    return VK_SUCCESS;
}

VkResult uploadDeviceBuffer(
    MemoryAllocator& allocator,
    CommandPool& commandPool,
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkDeviceSize alignment,
    VkBuffer& outBuffer,
    VkDeviceSize& outOffset) {
    if (size == 0 || data == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    freeBuffer(allocator, outBuffer, outOffset);

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void* stagingData = nullptr;
    if (createStagingBuffer(allocator, size, stagingBuffer, stagingOffset, &stagingData) != VK_SUCCESS || !stagingData) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    std::memcpy(stagingData, data, static_cast<size_t>(size));

    auto [buffer, offset] = allocator.allocate(
        size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        alignment);
    if (buffer == VK_NULL_HANDLE) {
        allocator.free(stagingBuffer, stagingOffset);
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    outBuffer = buffer;
    outOffset = offset;

    VkCommandBuffer cmd = commandPool.beginCommands();
    if (cmd == VK_NULL_HANDLE) {
        std::cerr << "[UPLOAD] beginCommands failed"
                  << " dstBuffer=" << outBuffer
                  << " dstOffset=" << outOffset
                  << " size=" << size
                  << " usage=" << usage
                  << std::endl;
        allocator.free(stagingBuffer, stagingOffset);
        allocator.free(outBuffer, outOffset);
        outBuffer = VK_NULL_HANDLE;
        outOffset = 0;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferCopy region{};
    region.srcOffset = stagingOffset;
    region.dstOffset = outOffset;
    region.size = size;
    vkCmdCopyBuffer(cmd, stagingBuffer, outBuffer, 1, &region);
    if (!commandPool.endCommands(cmd)) {
        std::cerr << "[UPLOAD] copy failed"
                  << " srcBuffer=" << stagingBuffer
                  << " srcOffset=" << region.srcOffset
                  << " dstBuffer=" << outBuffer
                  << " dstOffset=" << region.dstOffset
                  << " size=" << region.size
                  << " usage=" << usage
                  << std::endl;
        allocator.free(stagingBuffer, stagingOffset);
        allocator.free(outBuffer, outOffset);
        outBuffer = VK_NULL_HANDLE;
        outOffset = 0;
        return VK_ERROR_DEVICE_LOST;
    }

    allocator.free(stagingBuffer, stagingOffset);
    return VK_SUCCESS;
}

VkResult downloadDeviceBuffer(
    MemoryAllocator& allocator,
    CommandPool& commandPool,
    VkBuffer srcBuffer,
    VkDeviceSize srcOffset,
    VkDeviceSize size,
    void* dstData) {
    if (size == 0 || dstData == nullptr || srcBuffer == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void* stagingData = nullptr;
    if (createDownloadStagingBuffer(allocator, size, stagingBuffer, stagingOffset, &stagingData) != VK_SUCCESS || !stagingData) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    VkCommandBuffer cmd = commandPool.beginCommands();
    if (cmd == VK_NULL_HANDLE) {
        allocator.free(stagingBuffer, stagingOffset);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = stagingOffset;
    region.size = size;
    vkCmdCopyBuffer(cmd, srcBuffer, stagingBuffer, 1, &region);
    if (!commandPool.endCommands(cmd)) {
        std::cerr << "[DOWNLOAD] copy failed"
                  << " srcBuffer=" << srcBuffer
                  << " srcOffset=" << region.srcOffset
                  << " dstBuffer=" << stagingBuffer
                  << " dstOffset=" << region.dstOffset
                  << " size=" << region.size
                  << std::endl;
        allocator.free(stagingBuffer, stagingOffset);
        return VK_ERROR_DEVICE_LOST;
    }

    std::memcpy(dstData, stagingData, static_cast<size_t>(size));

    allocator.free(stagingBuffer, stagingOffset);
    return VK_SUCCESS;
}

void freeBuffer(MemoryAllocator& allocator, VkBuffer& buffer, VkDeviceSize& offset) {
    if (buffer != VK_NULL_HANDLE) {
        allocator.free(buffer, offset);
        buffer = VK_NULL_HANDLE;
        offset = 0;
    }
}

void freeBufferView(VulkanDevice& device, VkBufferView& bufferView) {
    if (bufferView != VK_NULL_HANDLE) {
        vkDestroyBufferView(device.getDevice(), bufferView, nullptr);
        bufferView = VK_NULL_HANDLE;
    }
}
