#pragma once

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanDevice;

class ContactSystemComputeStage {
public:
    explicit ContactSystemComputeStage(VulkanDevice& vulkanDevice);
    ~ContactSystemComputeStage();

    void dispatchGather(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, uint32_t elementCount) const;

    bool createDescriptorPool(uint32_t maxSets);
    bool createDescriptorSetLayout();
    bool createPipeline();

    bool createGatherDescriptorSet(
        VkBuffer sourceFieldBuffer,
        VkDeviceSize sourceFieldOffset,
        uint32_t sourceElementCount,
        VkBuffer destAccumulatorBuffer,
        VkDeviceSize destAccumulatorOffset,
        uint32_t destElementCount,
        VkBuffer edgesBuffer,
        VkDeviceSize edgesOffset,
        uint32_t edgeCount,
        VkBuffer indicesBuffer,
        VkDeviceSize indicesOffset,
        uint32_t indexCount,
        VkDescriptorSet& outSet) const;

    VkDescriptorPool getDescriptorPool() const { return descriptorPool; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    VkPipelineLayout getPipelineLayout() const { return pipelineLayout; }
    VkPipeline getPipeline() const { return pipeline; }

private:
    VulkanDevice& vulkanDevice;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};
