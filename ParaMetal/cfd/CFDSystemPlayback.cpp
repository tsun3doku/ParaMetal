#include "CFDSystemPlayback.hpp"

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"

#include <glm/glm.hpp>

CFDSystemPlayback::~CFDSystemPlayback() {
    cleanup();
}

bool CFDSystemPlayback::initialize(VulkanDevice& device, MemoryAllocator& alloc, uint32_t nodeCount_, uint32_t frameCapacity_) {
    cleanup();
    if (nodeCount_ == 0 || frameCapacity_ == 0) {
        return false;
    }

    VkDeviceSize alignment = device.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    VkDeviceSize rawStride = static_cast<VkDeviceSize>(nodeCount_) * sizeof(glm::vec4);
    VkDeviceSize alignedStride = (rawStride + alignment - 1) & ~(alignment - 1);

    VkDeviceSize totalSize = static_cast<VkDeviceSize>(frameCapacity_) * alignedStride;
    auto [buffer, offset] = alloc.allocate(
        totalSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        alignment);

    if (buffer == VK_NULL_HANDLE) {
        cleanup();
        return false;
    }

    allocator = &alloc;
    historyBuffer = buffer;
    historyBufferOffset = offset;
    nodeCount = nodeCount_;
    frameCapacity = frameCapacity_;
    recordedFrames = 0;
    frameStride = alignedStride;
    return true;
}

void CFDSystemPlayback::cleanup() {
    if (allocator && historyBuffer != VK_NULL_HANDLE) {
        allocator->free(historyBuffer, historyBufferOffset);
    }
    allocator = nullptr;
    historyBuffer = VK_NULL_HANDLE;
    historyBufferOffset = 0;
    nodeCount = 0;
    frameCapacity = 0;
    recordedFrames = 0;
    frameStride = 0;
}

void CFDSystemPlayback::recordFrame(VkCommandBuffer cmd, VkBuffer srcBuffer, VkDeviceSize srcOffset) {
    if (historyBuffer == VK_NULL_HANDLE || recordedFrames >= frameCapacity) {
        return;
    }

    VkDeviceSize dstOffset = historyBufferOffset + (static_cast<VkDeviceSize>(recordedFrames) * frameStride);
    VkDeviceSize copySize = static_cast<VkDeviceSize>(nodeCount) * sizeof(glm::vec4);
    VkBufferCopy region{srcOffset, dstOffset, copySize};
    vkCmdCopyBuffer(cmd, srcBuffer, historyBuffer, 1, &region);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // Replay copies frames back out as well as sampling them.
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier.buffer = historyBuffer;
    barrier.offset = dstOffset;
    barrier.size = copySize;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr);

    ++recordedFrames;
}

void CFDSystemPlayback::reset() {
    recordedFrames = 0;
}
