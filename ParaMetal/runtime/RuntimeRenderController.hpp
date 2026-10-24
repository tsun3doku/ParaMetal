#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

#include "render/RenderSettings.hpp"

class RenderRuntime;

struct RuntimeRenderFrameResult {
    bool submitted = false;
    uint32_t frameSlot = 0;
};

class RuntimeRenderController {
public:
    RuntimeRenderController(RenderRuntime& renderRuntime);

    RuntimeRenderFrameResult renderFrame(
        uint32_t& frameCounter,
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const app::RenderSettings& settings);

private:
    RenderRuntime& renderRuntime;
};
