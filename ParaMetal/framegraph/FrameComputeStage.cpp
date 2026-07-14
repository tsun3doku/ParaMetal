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

FrameComputeCollection FrameComputeStage::collect(uint32_t frameIndex, const std::vector<ComputePass*>& computePasses) {
    FrameComputeCollection collection{};

    if (computePasses.empty()) {
        return collection;
    }

    bool hasAnyComputeWrites = false;
    std::vector<VkCommandBuffer> submitCommandBuffers;

    for (ComputePass* computePass : computePasses) {
        if (!computePass) {
            continue;
        }

        computePass->update();

        if (computePass->hasDispatchableComputeWork()) {
            hasAnyComputeWrites = true;
            const auto& computeCommandBuffers = computePass->getComputeCommandBuffers();
            if (frameIndex >= computeCommandBuffers.size()) {
                collection.result = FrameStageResult::Fatal;
                return collection;
            }

            VkCommandBuffer computeCommandBuffer = computeCommandBuffers[frameIndex];
            vkResetCommandBuffer(computeCommandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
            computePass->recordComputeCommands(computeCommandBuffer, frameIndex);
            submitCommandBuffers.push_back(computeCommandBuffer);
            collection.synchronization = computePass->getSynchronization();
        }
    }

    if (!submitCommandBuffers.empty()) {
        collection.commandBuffer = submitCommandBuffers.front();
    }

    const bool queuesAreShared = vulkanDevice.getComputeQueue() == vulkanDevice.getGraphicsQueue();
    const bool computePresent = (collection.commandBuffer != VK_NULL_HANDLE);

    if (hasAnyComputeWrites) {
        collection.waitDstStageMask = frameGraphRuntime.getComputeToGraphicsWaitDstStageMask();
        collection.insertComputeToGraphicsBarrier = queuesAreShared && computePresent;
        collection.waitForComputeSemaphore = !queuesAreShared && computePresent;
        if (collection.waitForComputeSemaphore && collection.waitDstStageMask == 0) {
            collection.waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        }
    }

    return collection;
}
