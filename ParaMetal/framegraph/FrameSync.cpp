#include "FrameSync.hpp"
#include "ComputePass.hpp"

#include <iostream>

bool FrameSync::initialize(VkDevice deviceHandle, uint32_t maxFramesInFlight) {
    shutdown();

    device = deviceHandle;
    frameCount = maxFramesInFlight;
    currentFrame = 0;

    if (device == VK_NULL_HANDLE || frameCount == 0) {
        return false;
    }

    frameSlots.assign(frameCount, FrameSlot{});

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    for (uint32_t index = 0; index < frameCount; ++index) {
        FrameSlot& slot = frameSlots[index];
        if (vkCreateFence(device, &fenceInfo, nullptr, &slot.computeFence) != VK_SUCCESS) {
            shutdown();
            return false;
        }
    }

    return true;
}

void FrameSync::shutdown() {
    if (device != VK_NULL_HANDLE) {
        for (FrameSlot& slot : frameSlots) {
            if (slot.computeFence != VK_NULL_HANDLE) {
                vkDestroyFence(device, slot.computeFence, nullptr);
            }
        }
    }

    frameSlots.clear();
    frameCount = 0;
    currentFrame = 0;
    device = VK_NULL_HANDLE;
}

void FrameSync::prepareExternalFrame(uint32_t frameIndex) {
    if (frameCount == 0) {
        return;
    }

    currentFrame = frameIndex % frameCount;
    if (device == VK_NULL_HANDLE || currentFrame >= frameSlots.size()) {
        return;
    }

    FrameSlot& slot = frameSlots[currentFrame];
    slot.computePresentedThisLap = false;
    if (slot.computeSubmittedThisLap) {
        vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, UINT64_MAX);
        slot.computeSubmittedThisLap = false;
    }
}

bool FrameSync::wasComputePresentedThisLap() const {
    if (currentFrame >= frameSlots.size()) {
        return false;
    }
    return frameSlots[currentFrame].computePresentedThisLap;
}

void FrameSync::waitForAllFrameFences() {
    if (device == VK_NULL_HANDLE || frameSlots.empty()) {
        return;
    }

    for (const FrameSlot& slot : frameSlots) {
        // Skip unsignaled fences to avoid blocking indefinitely
        if (!slot.computeSubmittedThisLap) {
            continue;
        }
        vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, UINT64_MAX);
    }
}

VkResult FrameSync::submitComputeBatch(
    VkQueue computeQueue,
    const std::vector<FrameComputeSubmission>& submissions) {
    if (device == VK_NULL_HANDLE || currentFrame >= frameSlots.size() || computeQueue == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    if (submissions.empty()) {
        return VK_SUCCESS;
    }

    FrameSlot& slot = frameSlots[currentFrame];
    slot.computeSubmittedThisLap = false;

    vkResetFences(device, 1, &slot.computeFence);

    struct PerSubmitData {
        std::vector<VkSemaphore> waitSemaphores;
        std::vector<VkPipelineStageFlags> waitStages;
        std::vector<uint64_t> waitValues;

        std::vector<VkSemaphore> signalSemaphores;
        std::vector<uint64_t> signalValues;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        VkSubmitInfo submitInfo{};
    };

    std::vector<PerSubmitData> submitData(submissions.size());
    std::vector<VkSubmitInfo> submitInfos(submissions.size());

    for (size_t i = 0; i < submissions.size(); ++i) {
        const auto& sub = submissions[i];
        PerSubmitData& data = submitData[i];
        data.commandBuffer = sub.commandBuffer;

        data.waitSemaphores.reserve(sub.synchronization.waits.size());
        data.waitStages.reserve(sub.synchronization.waits.size());
        data.waitValues.reserve(sub.synchronization.waits.size());
        for (const ComputePass::TimelineWait& wait : sub.synchronization.waits) {
            if (wait.semaphore == VK_NULL_HANDLE) {
                continue;
            }
            data.waitSemaphores.push_back(wait.semaphore);
            data.waitStages.push_back(wait.stageMask);
            data.waitValues.push_back(wait.value);
        }

        data.signalSemaphores.reserve(sub.synchronization.signals.size());
        data.signalValues.reserve(sub.synchronization.signals.size());
        for (const ComputePass::TimelineSignal& signal : sub.synchronization.signals) {
            if (signal.semaphore == VK_NULL_HANDLE) {
                continue;
            }
            data.signalSemaphores.push_back(signal.semaphore);
            data.signalValues.push_back(signal.value);
        }

        data.timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        data.timelineInfo.waitSemaphoreValueCount = static_cast<uint32_t>(data.waitValues.size());
        data.timelineInfo.pWaitSemaphoreValues = data.waitValues.empty() ? nullptr : data.waitValues.data();
        data.timelineInfo.signalSemaphoreValueCount = static_cast<uint32_t>(data.signalValues.size());
        data.timelineInfo.pSignalSemaphoreValues = data.signalValues.empty() ? nullptr : data.signalValues.data();

        data.submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        if (data.timelineInfo.waitSemaphoreValueCount > 0 || data.timelineInfo.signalSemaphoreValueCount > 0) {
            data.submitInfo.pNext = &data.timelineInfo;
        }
        data.submitInfo.waitSemaphoreCount = static_cast<uint32_t>(data.waitSemaphores.size());
        data.submitInfo.pWaitSemaphores = data.waitSemaphores.empty() ? nullptr : data.waitSemaphores.data();
        data.submitInfo.pWaitDstStageMask = data.waitStages.empty() ? nullptr : data.waitStages.data();
        data.submitInfo.commandBufferCount = 1;
        data.submitInfo.pCommandBuffers = &data.commandBuffer;
        data.submitInfo.signalSemaphoreCount = static_cast<uint32_t>(data.signalSemaphores.size());
        data.submitInfo.pSignalSemaphores = data.signalSemaphores.empty() ? nullptr : data.signalSemaphores.data();

        submitInfos[i] = data.submitInfo;
    }

    const VkResult computeResult = vkQueueSubmit(
        computeQueue,
        static_cast<uint32_t>(submitInfos.size()),
        submitInfos.data(),
        slot.computeFence);

    if (computeResult == VK_SUCCESS) {
        slot.computeSubmittedThisLap = true;
        slot.computePresentedThisLap = true;
    } else {
        std::cerr << "[SUBMIT] COMPUTE BATCH failed VkResult=" << static_cast<int>(computeResult)
                  << " frame=" << currentFrame << std::endl;
        return computeResult;
    }

    return VK_SUCCESS;
}

