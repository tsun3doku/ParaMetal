#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "HeatSystemStageContext.hpp"

class HeatReceiverRuntime;

class HeatSystemSurfaceStage {
public:
    explicit HeatSystemSurfaceStage(const HeatSystemStageContext& stageContext);

    void refreshSurfaceDescriptors(uint32_t nodeCount);
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSetLayout();
    bool createPipeline();
    void dispatchSurfaceTemperatureUpdates(
        VkCommandBuffer commandBuffer,
        uint32_t nodeCount,
        const std::vector<std::unique_ptr<HeatReceiverRuntime>>& receivers,
        const std::vector<VkDescriptorSet>& surfaceComputeSetsA,
        const std::vector<VkDescriptorSet>& surfaceComputeSetsB,
        bool finalWritesBufferB) const;
    void dispatchSurfaceGradientUpdates(
        VkCommandBuffer commandBuffer,
        uint32_t nodeCount,
        const std::vector<std::unique_ptr<HeatReceiverRuntime>>& receivers,
        const std::vector<VkDescriptorSet>& surfaceGradientComputeSetsA,
        const std::vector<VkDescriptorSet>& surfaceGradientComputeSetsB,
        bool finalWritesBufferB) const;

private:
    HeatSystemStageContext context;
};
