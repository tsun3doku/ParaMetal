#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>

#include "HeatSystemPresets.hpp"
#include "HeatSystemStageContext.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "util/Structs.hpp"

class PointRenderer;
class HeatSystemSimRuntime;

class HeatSystemVoronoiStage {
public:
    explicit HeatSystemVoronoiStage(const HeatSystemStageContext& stageContext);

    void dispatchDiffusionSubstep(
        VkCommandBuffer commandBuffer,
        VkDescriptorSet descriptorSet,
        const heat::HeatModelPushConstant& pushConstant,
        uint32_t workGroupCount) const;
    void insertInterSubstepBarrier(
        VkCommandBuffer commandBuffer,
        int substepIndex,
        uint32_t numSubsteps) const;
    void insertFinalTemperatureBarrier(
        VkCommandBuffer commandBuffer,
        uint32_t numSubsteps,
        VkBuffer bufferA,
        VkDeviceSize offsetA,
        VkBuffer bufferB,
        VkDeviceSize offsetB) const;
    bool finalSubstepWritesBufferB(uint32_t numSubsteps) const;
    bool createDescriptorPool(uint32_t numModels);
    bool createDescriptorSetLayout();
    bool createPipeline();

    HeatSystemStageContext context;
};
