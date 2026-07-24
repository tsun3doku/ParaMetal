#include "VulkanDevice.hpp"

void VulkanDevice::importExternal(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkQueue graphicsQueue,
    uint32_t queueFamilyIndex) {
    cleanup();

    this->physicalDevice = physicalDevice;
    this->device = device;
    this->graphicsQueue = graphicsQueue;
    this->computeQueue = graphicsQueue;

    if (physicalDevice != VK_NULL_HANDLE) {
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        chooseDepthResolveMode();
    }

    queueFamilyIndices.graphicsFamily = queueFamilyIndex;
    queueFamilyIndices.graphicsAndComputeFamily = queueFamilyIndex;
}

void VulkanDevice::cleanup() {
    device = VK_NULL_HANDLE;
    graphicsQueue = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    physicalDeviceProperties = {};
    queueFamilyIndices = {};
    depthResolveMode = VK_RESOLVE_MODE_NONE;
}

void VulkanDevice::chooseDepthResolveMode() {
    if (physicalDevice == VK_NULL_HANDLE) {
        depthResolveMode = VK_RESOLVE_MODE_NONE;
        return;
    }

    VkPhysicalDeviceDepthStencilResolveProperties depthResolveProps{};
    depthResolveProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &depthResolveProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    if (depthResolveProps.supportedDepthResolveModes & VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) {
        depthResolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    } else if (depthResolveProps.supportedDepthResolveModes & VK_RESOLVE_MODE_MAX_BIT) {
        depthResolveMode = VK_RESOLVE_MODE_MAX_BIT;
    } else if (depthResolveProps.supportedDepthResolveModes & VK_RESOLVE_MODE_MIN_BIT) {
        depthResolveMode = VK_RESOLVE_MODE_MIN_BIT;
    } else {
        depthResolveMode = VK_RESOLVE_MODE_NONE;
    }
}

bool VulkanDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& outMemoryTypeIndex) const {
    outMemoryTypeIndex = UINT32_MAX;
    if (physicalDevice == VK_NULL_HANDLE) {
        return false;
    }

    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            outMemoryTypeIndex = i;
            return true;
        }
    }

    return false;
}

uint32_t VulkanDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    uint32_t memoryTypeIndex = UINT32_MAX;
    (void)findMemoryType(typeFilter, properties, memoryTypeIndex);
    return memoryTypeIndex;
}
