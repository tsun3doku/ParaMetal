#pragma once

#include <vulkan/vulkan.h>

#include "FramePass.hpp"
#include "scene/SceneView.hpp"

struct FrameState {
    uint32_t frameIndex = 0;
    VkExtent2D extent{};
    render::SceneView sceneView{};
    render::RenderFlags flags{};
};

struct FrameSyncState {
    bool insertComputeToGraphicsBarrier = false;
    VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
};
