#include "FrameSync.hpp"
#include "ComputePass.hpp"

#include <chrono>
#include <iostream>

static constexpr uint64_t FrameFenceTimeoutNs = 5'000'000'000ull;

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

VkResult FrameSync::prepareExternalFrame(uint32_t frameIndex) {
    if (frameCount == 0) {
        return VK_SUCCESS;
    }

    currentFrame = frameIndex % frameCount;
    if (device == VK_NULL_HANDLE || currentFrame >= frameSlots.size()) {
        return VK_SUCCESS;
    }

    FrameSlot& slot = frameSlots[currentFrame];
    if (slot.computeSubmittedThisLap) {
        const auto waitStart = std::chrono::steady_clock::now();
        const VkResult waitResult =
            vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, FrameFenceTimeoutNs);
        if (waitResult != VK_SUCCESS) {
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - waitStart).count();
            std::cerr << "[FrameSync] frame fence wait failed: frame=" << currentFrame
                      << " elapsedMs=" << elapsedMs
                      << " VkResult=" << static_cast<int>(waitResult) << std::endl;
            return waitResult;
        }
        slot.computeSubmittedThisLap = false;
    }

    return VK_SUCCESS;
}

bool FrameSync::wasComputePresentedThisLap() const {
    if (currentFrame >= frameSlots.size()) {
        return false;
    }
    return frameSlots[currentFrame].computeSubmittedThisLap;
}

bool FrameSync::waitForAllFrameFences(uint64_t timeoutNs) {
    if (device == VK_NULL_HANDLE || frameSlots.empty()) {
        return true;
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeoutNs);
    for (size_t index = 0; index < frameSlots.size(); ++index) {
        const FrameSlot& slot = frameSlots[index];
        if (!slot.computeSubmittedThisLap) {
            continue;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            std::cerr << "[FrameSync] drain fence wait timed out before frame=" << index << std::endl;
            return false;
        }
        const VkResult waitResult = vkWaitForFences(
            device, 1, &slot.computeFence, VK_TRUE, static_cast<uint64_t>(remaining));
        if (waitResult != VK_SUCCESS) {
            std::cerr << "[FrameSync] drain fence wait failed: frame=" << index
                      << " VkResult=" << static_cast<int>(waitResult) << std::endl;
            return false;
        }
    }

    return true;
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
    if (slot.computeSubmittedThisLap) {
        std::cerr << "[FrameSync] compute batch rejected: frame " << currentFrame
                  << " already has an outstanding submission" << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const VkResult resetResult = vkResetFences(device, 1, &slot.computeFence);
    if (resetResult != VK_SUCCESS) {
        std::cerr << "[FrameSync] compute fence reset failed: frame=" << currentFrame
                  << " VkResult=" << static_cast<int>(resetResult) << std::endl;
        return resetResult;
    }

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
    } else {
        std::cerr << "[SUBMIT] COMPUTE BATCH failed VkResult=" << static_cast<int>(computeResult)
                  << " frame=" << currentFrame << std::endl;
        return computeResult;
    }

    return VK_SUCCESS;
}

