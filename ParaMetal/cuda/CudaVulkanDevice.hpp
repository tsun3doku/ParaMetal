#pragma once

#include <vulkan/vulkan.h>

namespace cudaVulkan {

int findDevice(VkPhysicalDevice physicalDevice);

// True once a matching CUDA device has been located in this process
bool deviceWasFound();

} // namespace cudaVulkan
