#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

#include "HeatSystemPresets.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "util/Structs.hpp"

class PointRenderer;
class HeatSystemSimRuntime;
class VulkanDevice;

class HeatSystemDiffusionStage {
public:
    explicit HeatSystemDiffusionStage(VulkanDevice& device);
    ~HeatSystemDiffusionStage();

    void dispatchDiffusionSubstep(
        VkCommandBuffer commandBuffer,
        VkDescriptorSet descriptorSet,
        const heat::HeatModelPushConstant& pushConstant,
        uint32_t workGroupCount) const;
    void insertFinalTemperatureBarrier(
        VkCommandBuffer commandBuffer,
        uint32_t numSubsteps,
        VkBuffer bufferA,
        VkDeviceSize offsetA,
        VkBuffer bufferB,
        VkDeviceSize offsetB,
        VkDeviceSize bufferSize) const;
    bool finalSubstepWritesBufferB(uint32_t numSubsteps) const;
    bool createDescriptorPool(uint32_t numModels);
    bool createDescriptorSetLayout();
    bool createPipeline();

    VkDescriptorPool getDescriptorPool() const { return descriptorPool; }
    VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

private:
    VulkanDevice& vulkanDevice;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};
