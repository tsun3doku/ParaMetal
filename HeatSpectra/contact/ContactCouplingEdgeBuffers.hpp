#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

struct ContactCouplingEdgeBuffers {
    uint32_t modelARuntimeModelId = 0;
    uint32_t modelBRuntimeModelId = 0;

    VkBuffer edgesAToB = VK_NULL_HANDLE;
    VkDeviceSize edgesAToBOffset = 0;
    uint32_t edgeCountAToB = 0;
    VkBuffer edgeIndexAToB = VK_NULL_HANDLE;
    VkDeviceSize edgeIndexAToBOffset = 0;

    VkBuffer edgesBToA = VK_NULL_HANDLE;
    VkDeviceSize edgesBToAOffset = 0;
    uint32_t edgeCountBToA = 0;
    VkBuffer edgeIndexBToA = VK_NULL_HANDLE;
    VkDeviceSize edgeIndexBToAOffset = 0;
};
