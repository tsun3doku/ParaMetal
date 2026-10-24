#pragma once

#include <vulkan/vulkan.h>

class VulkanDevice;

class VulkanExternalBuffer {
public:
    VulkanExternalBuffer() = default;
    ~VulkanExternalBuffer();

    VulkanExternalBuffer(const VulkanExternalBuffer&) = delete;
    VulkanExternalBuffer& operator=(const VulkanExternalBuffer&) = delete;

    bool initialize(VulkanDevice& vulkanDevice, VkDeviceSize size, VkBufferUsageFlags usage);
    void cleanup();

    void* exportWin32Handle() const;

    VkBuffer getBuffer() const { return buffer; }
    VkDeviceMemory getMemory() const { return memory; }
    VkDevice getDevice() const { return device; }
    VkDeviceSize getSize() const { return bufferSize; }
    VkDeviceSize getBufferSize() const { return bufferSize; }
    VkDeviceSize getAllocationSize() const { return allocationSize; }
    bool isValid() const { return buffer != VK_NULL_HANDLE && memory != VK_NULL_HANDLE; }

private:
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bufferSize = 0;
    VkDeviceSize allocationSize = 0;
};
