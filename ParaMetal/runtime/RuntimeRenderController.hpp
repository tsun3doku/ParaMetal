#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

#include "render/RenderSettings.hpp"

class RenderRuntime;
class FrameSync;
class MemoryAllocator;
class HeatSystemComputeController;

struct RuntimeRenderFrameResult {
    bool submitted = false;
    uint32_t frameSlot = 0;
};

class RuntimeRenderController {
public:
    RuntimeRenderController(RenderRuntime& renderRuntime, FrameSync& frameSync, MemoryAllocator* memoryAllocator, HeatSystemComputeController* heatSystemComputeController);

    RuntimeRenderFrameResult renderFrame(
        bool allowHeatSolve,
        uint32_t& frameCounter,
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const app::RenderSettings& settings);

private:
    RenderRuntime& renderRuntime;
    FrameSync& frameSync;
    MemoryAllocator* memoryAllocator = nullptr;
    HeatSystemComputeController* heatSystemComputeController = nullptr;
};
