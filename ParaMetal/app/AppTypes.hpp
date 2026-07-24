#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

struct AppVulkanContext {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;
    VkImage viewportImage = VK_NULL_HANDLE;
    VkFormat viewportFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D viewportExtent{};
};
