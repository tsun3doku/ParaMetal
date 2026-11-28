#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

class VulkanDevice;

class PickReadback {
public:
    explicit PickReadback(VulkanDevice& device);
    ~PickReadback();

    PickReadback(const PickReadback&) = delete;
    PickReadback& operator=(const PickReadback&) = delete;

    bool isInitialized() const { return initialized; }
    bool readPixel(VkImage image, int x, int y, uint32_t& outValue);
    void cleanup();

private:
    bool createCommandPool();
    bool createStagingBuffer();

    VulkanDevice& vulkanDevice;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
    void* stagingBufferMapped = nullptr;
    bool initialized = false;
};
