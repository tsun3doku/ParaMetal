#pragma once

#include <cstdint>
#include <vector>

#include "FrameTypes.hpp"

class VulkanDevice;
class FrameSync;
class ComputePass;
class VkFrameGraphRuntime;

// Collected compute work for a frame. Stages record into command buffers but
// never submit; FrameController assembles these into a FrameSubmission and
// commits via FrameSync::submitFrame.
struct FrameComputeCollection {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;  // VK_NULL_HANDLE = no compute this frame
    bool waitForComputeSemaphore = false;
    bool insertComputeToGraphicsBarrier = false;
    VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    FrameStageResult result = FrameStageResult::Continue;
};

class FrameComputeStage {
public:
    FrameComputeStage(VulkanDevice& vulkanDevice, VkFrameGraphRuntime& frameGraphRuntime, FrameSync& frameSync);

    // Records compute command buffers for the frame. Does NOT submit. Returns
    // the collected work; result indicates whether to proceed, recreate, etc.
    FrameComputeCollection collect(uint32_t frameIndex, const std::vector<ComputePass*>& computePasses);

private:
    VulkanDevice& vulkanDevice;
    VkFrameGraphRuntime& frameGraphRuntime;
    FrameSync& frameSync;
};
