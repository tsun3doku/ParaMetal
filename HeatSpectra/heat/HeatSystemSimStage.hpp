#pragma once

#include "HeatSystemSimRuntime.hpp"
#include "HeatSystemStageContext.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "util/Structs.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class HeatModelRuntime;
class HeatSystemSurfaceStage;
class HeatSystemVoronoiStage;
class ContactSystemComputeStage;
class HeatContactRuntime;

class HeatSystemSimStage {
public:
    explicit HeatSystemSimStage(const HeatSystemStageContext& stageContext);

    void recordComputeCommands(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        const HeatSystemSimRuntime& simRuntime,
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
        const HeatSystemVoronoiStage& voronoiStage,
        const HeatSystemSurfaceStage& surfaceStage,
        const ContactSystemComputeStage& contactStage,
        const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
        uint32_t maxNodeNeighbors,
        uint32_t numSubsteps) const;

private:
    HeatSystemStageContext context;
};
