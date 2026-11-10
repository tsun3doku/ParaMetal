#pragma once

#include <vulkan/vulkan.h>

namespace cudaVulkan {

int findDevice(VkPhysicalDevice physicalDevice);

// True once a matching CUDA device has been located in this process
bool deviceWasFound();

// Blocks until neither Vulkan nor CUDA has work in flight
bool waitForGpuIdle(VkDevice device, VkPhysicalDevice physicalDevice);

} // namespace cudaVulkan
