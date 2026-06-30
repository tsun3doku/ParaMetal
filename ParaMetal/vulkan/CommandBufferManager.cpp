#include "CommandBufferManager.hpp"
#include <stdexcept>
#include <iostream>

std::mutex CommandPool::queueSubmitMutex;

CommandPool::CommandPool(VulkanDevice& device, const char* name) 
    : vulkanDevice(device), debugName(name) {
    
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = device.getQueueFamilyIndices().graphicsFamily.value();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; 
    
    pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device.getDevice(), &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed to create command pool: " << debugName << std::endl;
        return;
    }
    
}

CommandPool::~CommandPool() {
    if (pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vulkanDevice.getDevice(), pool, nullptr);
    }
}

VkCommandBuffer CommandPool::beginCommands() {
    std::lock_guard<std::mutex> lock(poolMutex);  
    
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(vulkanDevice.getDevice(), &allocInfo, &commandBuffer) != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed to allocate command buffer from: " << debugName << std::endl;
        return VK_NULL_HANDLE;
    }
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    return commandBuffer;
}

bool CommandPool::endCommands(VkCommandBuffer commandBuffer) {
    if (commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    const VkResult endResult = vkEndCommandBuffer(commandBuffer);
    if (endResult != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed to end command buffer from: " << debugName
                  << " VkResult=" << static_cast<int>(endResult) << std::endl;
        std::lock_guard<std::mutex> poolLock(poolMutex);
        vkFreeCommandBuffers(vulkanDevice.getDevice(), pool, 1, &commandBuffer);
        return false;
    }
    
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    const VkResult fenceResult = vkCreateFence(vulkanDevice.getDevice(), &fenceInfo, nullptr, &fence);
    if (fenceResult != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed to create one-shot fence for: " << debugName
                  << " VkResult=" << static_cast<int>(fenceResult) << std::endl;
        std::lock_guard<std::mutex> poolLock(poolMutex);
        vkFreeCommandBuffers(vulkanDevice.getDevice(), pool, 1, &commandBuffer);
        return false;
    }
    
    VkResult submitResult = VK_SUCCESS;
    {
        std::lock_guard<std::mutex> queueLock(queueSubmitMutex);
        submitResult = vkQueueSubmit(vulkanDevice.getGraphicsQueue(), 1, &submitInfo, fence);
    }

    if (submitResult != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed to submit one-shot command buffer from: " << debugName
                  << " VkResult=" << static_cast<int>(submitResult)
                  << "; leaving command buffer allocated because GPU ownership was not proven."
                  << std::endl;
        vkDestroyFence(vulkanDevice.getDevice(), fence, nullptr);
        return false;
    }

    const VkResult waitResult = vkWaitForFences(vulkanDevice.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX);
    if (waitResult != VK_SUCCESS) {
        std::cerr << "[CommandPool] Failed waiting for one-shot command buffer from: " << debugName
                  << " VkResult=" << static_cast<int>(waitResult)
                  << "; leaving command buffer/fence allocated because GPU completion was not proven."
                  << std::endl;
        return false;
    }

    vkDestroyFence(vulkanDevice.getDevice(), fence, nullptr);

    std::lock_guard<std::mutex> poolLock(poolMutex);
    vkFreeCommandBuffers(vulkanDevice.getDevice(), pool, 1, &commandBuffer);
    return true;
}

void CommandPool::copyBuffer(VkBuffer srcBuffer, VkDeviceSize srcOffset, 
                              VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size) {
    VkCommandBuffer cmdBuffer = beginCommands();
    
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = srcOffset;
    copyRegion.dstOffset = dstOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(cmdBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    
    endCommands(cmdBuffer);
}

void CommandPool::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginCommands();
    
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    endCommands(commandBuffer);
}

void CommandPool::transitionImageLayout(VkImage image, VkFormat format, 
                                        VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginCommands();
    
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition!");
    }
    
    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    endCommands(commandBuffer);
}
