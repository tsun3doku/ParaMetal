#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class HeatModelRuntime;
class HeatSystemDiffusionStage;
class ContactSystemComputeStage;
class HeatContactRuntime;

class HeatSystemSimStage {
public:
    explicit HeatSystemSimStage() = default;

    void recordComputeCommands(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemDiffusionStage& diffusionStage,
        const ContactSystemComputeStage& contactStage,
        const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
        bool steppingPhysics,
        bool captureFrame,
        uint32_t numSubsteps) const;

private:
    void recordSim(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemDiffusionStage& diffusionStage,
        const ContactSystemComputeStage& contactStage,
        const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
        uint32_t numSubsteps) const;

    void recordHistoryCapture(
        VkCommandBuffer commandBuffer,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemDiffusionStage& diffusionStage,
        uint32_t numSubsteps) const;
};
