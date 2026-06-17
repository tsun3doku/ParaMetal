#include "HeatSystemPlayback.hpp"

#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"

HeatSystemPlayback::~HeatSystemPlayback() {
    cleanup();
}

bool HeatSystemPlayback::initialize(VulkanDevice& device, MemoryAllocator& alloc, uint32_t nodeCount_, uint32_t frameCapacity_) {
    cleanup();
    if (nodeCount_ == 0 || frameCapacity_ == 0) {
        return false;
    }

    VkDeviceSize totalSize = static_cast<VkDeviceSize>(frameCapacity_) * static_cast<VkDeviceSize>(nodeCount_) * sizeof(float);
    auto [buffer, offset] = alloc.allocate(
        totalSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        device.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment);

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
    frameStride = static_cast<VkDeviceSize>(nodeCount_) * sizeof(float);
    return true;
}

void HeatSystemPlayback::cleanup() {
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

void HeatSystemPlayback::recordFrame(VkCommandBuffer cmd, VkBuffer srcTempBuffer, VkDeviceSize srcOffset) {
    if (historyBuffer == VK_NULL_HANDLE || recordedFrames >= frameCapacity) {
        return;
    }

    VkDeviceSize dstOffset = historyBufferOffset + (static_cast<VkDeviceSize>(recordedFrames) * frameStride);
    VkBufferCopy region{srcOffset, dstOffset, frameStride};
    vkCmdCopyBuffer(cmd, srcTempBuffer, historyBuffer, 1, &region);

    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.buffer = historyBuffer;
    barrier.offset = dstOffset;
    barrier.size = frameStride;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr);

    ++recordedFrames;
}

void HeatSystemPlayback::reset() {
    recordedFrames = 0;
}
