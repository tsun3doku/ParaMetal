#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "HeatSystemStageContext.hpp"

class HeatModelRuntime;

class HeatSystemSurfaceStage {
public:
    explicit HeatSystemSurfaceStage(const HeatSystemStageContext& stageContext);

    bool createDescriptorPool(uint32_t numModels);
    bool createDescriptorSetLayout();
    bool createPipeline();
    void dispatchSurfaceTemperatureUpdates(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool finalWritesBufferB) const;

    void dispatchSurfaceGradientUpdates(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool finalWritesBufferB) const;

private:
    void dispatchSurfacePass(
        VkCommandBuffer commandBuffer,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool finalWritesBufferB,
        bool isGradientPass) const;

    HeatSystemStageContext context;
};
