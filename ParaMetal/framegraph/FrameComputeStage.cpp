#include "FrameComputeStage.hpp"

#include "ComputePass.hpp"
#include "FrameSync.hpp"
#include "VkFrameGraphRuntime.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "heat/HeatSystem.hpp"
#include "render/RenderConfig.hpp"

FrameComputeStage::FrameComputeStage(VulkanDevice& vulkanDevice, VkFrameGraphRuntime& frameGraphRuntime, FrameSync& frameSync)
    : vulkanDevice(vulkanDevice),
      frameGraphRuntime(frameGraphRuntime),
      frameSync(frameSync) {
    timestampPeriod = vulkanDevice.getPhysicalDeviceProperties().limits.timestampPeriod;
    timingValid.assign(renderconfig::MaxFramesInFlight, 0);

    if (timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo queryInfo{};
        queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = renderconfig::MaxFramesInFlight * 2;
        if (vkCreateQueryPool(vulkanDevice.getDevice(), &queryInfo, nullptr, &timingQueryPool) != VK_SUCCESS) {
            timingQueryPool = VK_NULL_HANDLE;
        }
    }
}

FrameComputeStage::~FrameComputeStage() {
    if (timingQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(vulkanDevice.getDevice(), timingQueryPool, nullptr);
    }
}

FrameComputeCollection FrameComputeStage::collect(uint32_t frameIndex, const std::vector<ComputePass*>& computePasses) {
    FrameComputeCollection collection{};

    const uint32_t queryBase = frameIndex * 2;
    if (timingQueryPool != VK_NULL_HANDLE &&
        frameIndex < timingValid.size() && timingValid[frameIndex] != 0) {
        uint64_t timestamps[2]{};
        const VkResult result = vkGetQueryPoolResults(
            vulkanDevice.getDevice(), timingQueryPool, queryBase, 2,
            sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result == VK_SUCCESS && timestamps[1] > timestamps[0]) {
            collection.computeGpuMs = static_cast<float>(timestamps[1] - timestamps[0]) * timestampPeriod * 1e-6f;
        }
    }

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
            if (timingQueryPool != VK_NULL_HANDLE) {
                vkResetQueryPool(vulkanDevice.getDevice(), timingQueryPool, queryBase, 2);
                computePass->setComputeTimingQueries(timingQueryPool, queryBase, queryBase + 1);
            }
            computePass->recordComputeCommands(computeCommandBuffer, frameIndex);
            submitCommandBuffers.push_back(computeCommandBuffer);
            collection.synchronization = computePass->getSynchronization();
            if (timingQueryPool != VK_NULL_HANDLE && frameIndex < timingValid.size()) {
                timingValid[frameIndex] = 1;
            }
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
