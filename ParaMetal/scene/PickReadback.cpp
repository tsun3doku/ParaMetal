#include "PickReadback.hpp"

#include "vulkan/VulkanDevice.hpp"

#include <iostream>

PickReadback::PickReadback(VulkanDevice& device)
    : vulkanDevice(device) {
    if (!createCommandPool() || !createStagingBuffer()) {
        std::cerr << "[PickReadback] Initialization failed" << std::endl;
        cleanup();
        return;
    }
    initialized = true;
}

PickReadback::~PickReadback() {
    cleanup();
}

bool PickReadback::createCommandPool() {
    const QueueFamilyIndices indices = vulkanDevice.getQueueFamilyIndices();
    if (!indices.graphicsAndComputeFamily.has_value()) {
        return false;
    }

    VkCommandPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = indices.graphicsAndComputeFamily.value();
    info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    return vkCreateCommandPool(vulkanDevice.getDevice(), &info, nullptr, &commandPool) == VK_SUCCESS;
}

bool PickReadback::createStagingBuffer() {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vulkanDevice.getDevice(), &info, nullptr, &stagingBuffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(vulkanDevice.getDevice(), stagingBuffer, &requirements);

    uint32_t memoryTypeIndex = UINT32_MAX;
    if (!vulkanDevice.findMemoryType(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memoryTypeIndex)) {
        return false;
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(vulkanDevice.getDevice(), &alloc, nullptr, &stagingBufferMemory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(vulkanDevice.getDevice(), stagingBuffer, stagingBufferMemory, 0) != VK_SUCCESS) {
        return false;
    }
    return vkMapMemory(
        vulkanDevice.getDevice(),
        stagingBufferMemory,
        0,
        sizeof(uint32_t),
        0,
        &stagingBufferMapped) == VK_SUCCESS;
}

bool PickReadback::readPixel(VkImage image, int x, int y, uint32_t& outValue) {
    if (!initialized || image == VK_NULL_HANDLE || x < 0 || y < 0) {
        return false;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vulkanDevice.getDevice(), &allocInfo, &commandBuffer) != VK_SUCCESS) {
        return false;
    }

    const auto freeCommandBuffer = [&]() {
        vkFreeCommandBuffers(vulkanDevice.getDevice(), commandPool, 1, &commandBuffer);
    };

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &begin) != VK_SUCCESS) {
        freeCommandBuffer();
        return false;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {x, y, 0};
    region.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(
        commandBuffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        stagingBuffer,
        1,
        &region);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        freeCommandBuffer();
        return false;
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(vulkanDevice.getDevice(), &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        freeCommandBuffer();
        return false;
    }

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(vulkanDevice.getGraphicsQueue(), 1, &submit, fence) != VK_SUCCESS) {
        vkDestroyFence(vulkanDevice.getDevice(), fence, nullptr);
        freeCommandBuffer();
        return false;
    }

    constexpr uint64_t timeoutNs = 10ull * 1000ull * 1000ull * 1000ull;
    const VkResult waitResult = vkWaitForFences(
        vulkanDevice.getDevice(), 1, &fence, VK_TRUE, timeoutNs);
    vkDestroyFence(vulkanDevice.getDevice(), fence, nullptr);
    freeCommandBuffer();
    if (waitResult != VK_SUCCESS) {
        return false;
    }

    outValue = *static_cast<uint32_t*>(stagingBufferMapped);
    return true;
}

void PickReadback::cleanup() {
    if (stagingBufferMapped != nullptr) {
        vkUnmapMemory(vulkanDevice.getDevice(), stagingBufferMemory);
        stagingBufferMapped = nullptr;
    }
    if (stagingBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(vulkanDevice.getDevice(), stagingBuffer, nullptr);
        stagingBuffer = VK_NULL_HANDLE;
    }
    if (stagingBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(vulkanDevice.getDevice(), stagingBufferMemory, nullptr);
        stagingBufferMemory = VK_NULL_HANDLE;
    }
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vulkanDevice.getDevice(), commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }
    initialized = false;
}
