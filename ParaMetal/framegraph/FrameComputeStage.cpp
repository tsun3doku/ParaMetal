#include "FrameComputeStage.hpp"

#include "ComputePass.hpp"
#include "FrameSync.hpp"
#include "VkFrameGraphRuntime.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "heat/HeatSystem.hpp"

FrameComputeStage::FrameComputeStage(VulkanDevice& vulkanDevice, VkFrameGraphRuntime& frameGraphRuntime, FrameSync& frameSync)
    : vulkanDevice(vulkanDevice),
      frameGraphRuntime(frameGraphRuntime),
      frameSync(frameSync) {
}

FrameStageResult FrameComputeStage::execute(uint32_t frameIndex, const std::vector<ComputePass*>& computePasses, FrameSyncState& syncState) {
    syncState = {};
    if (computePasses.empty()) {
        return FrameStageResult::Continue;
    }

    bool hasAnyComputeWrites = false;
    std::vector<VkCommandBuffer> submitCommandBuffers;

    HeatSystem* heatSystemToDebug = nullptr;

    for (ComputePass* computePass : computePasses) {
        if (!computePass) {
            continue;
        }

        computePass->update();

        if (computePass->hasDispatchableComputeWork()) {
            hasAnyComputeWrites = true;
            if (auto* hs = dynamic_cast<HeatSystem*>(computePass)) {
                heatSystemToDebug = hs;
            }
            const auto& computeCommandBuffers = computePass->getComputeCommandBuffers();
            if (frameIndex >= computeCommandBuffers.size()) {
                syncState = {};
                return FrameStageResult::Fatal;
            }

            VkCommandBuffer computeCommandBuffer = computeCommandBuffers[frameIndex];
            vkResetCommandBuffer(computeCommandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
            computePass->recordComputeCommands(computeCommandBuffer, frameIndex);
            submitCommandBuffers.push_back(computeCommandBuffer);
        }
    }

    const bool queuesAreShared = vulkanDevice.getComputeQueue() == vulkanDevice.getGraphicsQueue();
    bool computeSubmittedThisFrame = false;

    if (!submitCommandBuffers.empty()) {
        frameSync.prepareComputeSubmit();

        for (VkCommandBuffer commandBuffer : submitCommandBuffers) {
            const VkResult computeSubmitResult = frameSync.submitCompute(vulkanDevice.getComputeQueue(), commandBuffer, !queuesAreShared);

            if (computeSubmitResult != VK_SUCCESS) {
                if (computeSubmitResult == VK_ERROR_DEVICE_LOST) {
                    syncState = {};
                    return FrameStageResult::Fatal;
                }
                syncState = {};
                return FrameStageResult::RecreateSwapchain;
            }
        }

        computeSubmittedThisFrame = true;
    }



    if (hasAnyComputeWrites) {
        syncState.waitDstStageMask = frameGraphRuntime.getComputeToGraphicsWaitDstStageMask();
        syncState.insertComputeToGraphicsBarrier = queuesAreShared && computeSubmittedThisFrame;
        syncState.waitForComputeSemaphore = !queuesAreShared && computeSubmittedThisFrame;
        if (syncState.waitForComputeSemaphore && syncState.waitDstStageMask == 0) {
            syncState.waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
    }

    return FrameStageResult::Continue;
}