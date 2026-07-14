#include "VulkanDevice.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

void VulkanDevice::init(
    VkInstance instance,
    VkSurfaceKHR surface,
    const std::vector<const char*>& deviceExtensions,
    const std::vector<const char*>& validationLayers,
    bool enableValidationLayers) {
    cleanup();

    this->surface = surface;
    this->deviceExtensions = deviceExtensions;
    this->validationLayers = validationLayers;
    this->enableValidationLayers = enableValidationLayers;

    if (instance == VK_NULL_HANDLE || surface == VK_NULL_HANDLE) {
        throw std::runtime_error("VulkanDevice::init received invalid instance or surface");
    }

    pickPhysicalDevice(instance, surface);
    createLogicalDevice(surface);
    chooseDepthResolveMode();
    ownsDevice = true;
}

void VulkanDevice::importExternal(
    VkPhysicalDevice physicalDevice,
    VkDevice device,
    VkQueue graphicsQueue,
    uint32_t queueFamilyIndex,
    VkSurfaceKHR surface) {
    cleanup();

    this->physicalDevice = physicalDevice;
    this->device = device;
    this->surface = surface;
    this->graphicsQueue = graphicsQueue;
    this->computeQueue = graphicsQueue;
    this->presentQueue = graphicsQueue;

    if (physicalDevice != VK_NULL_HANDLE) {
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
        chooseDepthResolveMode();
    }

    queueFamilyIndices.graphicsFamily = queueFamilyIndex;
    queueFamilyIndices.graphicsAndComputeFamily = queueFamilyIndex;
    queueFamilyIndices.presentFamily = queueFamilyIndex;
    ownsDevice = false;
}

void VulkanDevice::cleanup() {
    if (ownsDevice && device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }

    device = VK_NULL_HANDLE;
    graphicsQueue = VK_NULL_HANDLE;
    computeQueue = VK_NULL_HANDLE;
    presentQueue = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    surface = VK_NULL_HANDLE;
    physicalDeviceProperties = {};
    queueFamilyIndices = {};
    depthResolveMode = VK_RESOLVE_MODE_NONE;
    deviceExtensions.clear();
    validationLayers.clear();
    enableValidationLayers = false;
    ownsDevice = false;
}

static int scoreDeviceType(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:     return 1000;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:    return 100;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:       return 50;
        default:                                        return 0;
    }
}

void VulkanDevice::pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    int bestScore = -1;
    for (const VkPhysicalDevice candidate : devices) {
        if (!isDeviceSuitable(candidate, surface)) {
            continue;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);

        const int score = scoreDeviceType(props.deviceType);
        if (score > bestScore) {
            bestScore = score;
            physicalDevice = candidate;
            physicalDeviceProperties = props;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find a suitable GPU");
    }
}

void VulkanDevice::createLogicalDevice(VkSurfaceKHR surface) {
    queueFamilyIndices = findQueueFamilies(physicalDevice, surface);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {
        queueFamilyIndices.graphicsAndComputeFamily.value(),
        queueFamilyIndices.presentFamily.value()
    };

    const float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceVulkan12Features supportedVulkan12Features{};
    supportedVulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;

    VkPhysicalDeviceFeatures2 supportedFeatures2{};
    supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    supportedFeatures2.pNext = &supportedVulkan12Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &supportedFeatures2);

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.descriptorIndexing = supportedVulkan12Features.descriptorIndexing;
    vulkan12Features.descriptorBindingUniformBufferUpdateAfterBind = supportedVulkan12Features.descriptorBindingUniformBufferUpdateAfterBind;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = supportedVulkan12Features.descriptorBindingStorageBufferUpdateAfterBind;
    vulkan12Features.descriptorBindingUpdateUnusedWhilePending = supportedVulkan12Features.descriptorBindingUpdateUnusedWhilePending;
    vulkan12Features.timelineSemaphore = supportedVulkan12Features.timelineSemaphore;

    if (!supportedVulkan12Features.timelineSemaphore) {
        throw std::runtime_error("Vulkan timeline semaphores are required for CUDA interop");
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.sampleRateShading = supportedFeatures2.features.sampleRateShading;
    deviceFeatures.samplerAnisotropy = supportedFeatures2.features.samplerAnisotropy;
    deviceFeatures.wideLines = supportedFeatures2.features.wideLines;
    deviceFeatures.fillModeNonSolid = supportedFeatures2.features.fillModeNonSolid;
    deviceFeatures.independentBlend = supportedFeatures2.features.independentBlend;
    deviceFeatures.geometryShader = supportedFeatures2.features.geometryShader;
    deviceFeatures.shaderFloat64 = supportedFeatures2.features.shaderFloat64;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan12Features;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device");
    }

    vkGetDeviceQueue(device, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilyIndices.graphicsAndComputeFamily.value(), 0, &computeQueue);
    vkGetDeviceQueue(device, queueFamilyIndices.presentFamily.value(), 0, &presentQueue);
}

SwapChainSupportDetails VulkanDevice::querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) const {
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount > 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

bool VulkanDevice::isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    const QueueFamilyIndices indices = findQueueFamilies(device, surface);

    const bool extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;

    if (extensionsSupported) {
        const SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}

bool VulkanDevice::checkDeviceExtensionSupport(VkPhysicalDevice device) const {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices VulkanDevice::findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) const {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.graphicsAndComputeFamily = i;
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, static_cast<uint32_t>(i), surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        ++i;
    }

    return indices;
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
