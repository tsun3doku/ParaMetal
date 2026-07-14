#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class HeatModelRuntime;
class HeatSystemDiffusionStage;

class HeatSystemSimStage {
public:
    explicit HeatSystemSimStage() = default;

    // Records the physics substeps and (optionally) a history capture frame.
    // temperatureBufferAIsCurrent is the persistent read-buffer parity carried
    // across frames; the returned value is the parity after all substeps.
    bool recordComputeCommands(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemDiffusionStage& diffusionStage,
        bool steppingPhysics,
        bool captureFrame,
        bool temperatureBufferAIsCurrent,
        uint32_t diffusionSubsteps) const;

private:
    bool recordSim(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemDiffusionStage& diffusionStage,
        bool temperatureBufferAIsCurrent,
        uint32_t diffusionSubsteps) const;

    void recordHistoryCapture(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        bool finalWritesBufferB) const;
};
