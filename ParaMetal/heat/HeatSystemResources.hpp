#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class HeatSystemResources {
public:
    HeatSystemResources() = default;
    ~HeatSystemResources() = default;

    VkDescriptorPool voronoiDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout voronoiDescriptorSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout voronoiPipelineLayout = VK_NULL_HANDLE;
    VkPipeline voronoiPipeline = VK_NULL_HANDLE;

    VkDescriptorPool surfaceDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout surfaceDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout surfaceGradientDescriptorSetLayout = VK_NULL_HANDLE;

    VkPipelineLayout surfacePipelineLayout = VK_NULL_HANDLE;
    VkPipeline surfacePipeline = VK_NULL_HANDLE;
    VkPipelineLayout surfaceGradientPipelineLayout = VK_NULL_HANDLE;
    VkPipeline surfaceGradientPipeline = VK_NULL_HANDLE;

    bool hasContact = false;
};
