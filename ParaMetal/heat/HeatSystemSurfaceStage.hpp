#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class HeatModelRuntime;
class VulkanDevice;

class HeatSystemSurfaceStage {
public:
    explicit HeatSystemSurfaceStage(VulkanDevice& device);
    ~HeatSystemSurfaceStage();

    bool createDescriptorPool(uint32_t numModels);
    bool createDescriptorSetLayout();
    bool createPipeline();
    void dispatchSurfaceTemperatureUpdates(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool replayFromHistory,
        bool finalWritesBufferB,
        uint32_t currentFrame) const;

    void dispatchSurfaceGradientUpdates(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool replayFromHistory,
        bool finalWritesBufferB,
        uint32_t currentFrame) const;

    VkDescriptorPool getDescriptorPool() const { return descriptorPool; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    VkDescriptorSetLayout getGradientDescriptorSetLayout() const { return gradientDescriptorSetLayout; }

private:
    void dispatchSurfacePass(
        VkCommandBuffer commandBuffer,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool replayFromHistory,
        bool finalWritesBufferB,
        bool isGradientPass,
        uint32_t currentFrame) const;

    VulkanDevice& vulkanDevice;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout gradientDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout gradientPipelineLayout = VK_NULL_HANDLE;
    VkPipeline gradientPipeline = VK_NULL_HANDLE;
};
