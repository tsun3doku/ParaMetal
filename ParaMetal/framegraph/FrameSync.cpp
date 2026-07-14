#include "FrameSync.hpp"

#include <array>
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

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    const auto failCreate = [this]() {
        shutdown();
        return false;
    };

    for (uint32_t index = 0; index < frameCount; ++index) {
        FrameSlot& slot = frameSlots[index];
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &slot.imageAvailableSemaphore) != VK_SUCCESS) {
            return failCreate();
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &slot.renderFinishedSemaphore) != VK_SUCCESS) {
            return failCreate();
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &slot.computeFinishedSemaphore) != VK_SUCCESS) {
            return failCreate();
        }
        if (vkCreateFence(device, &fenceInfo, nullptr, &slot.graphicsFence) != VK_SUCCESS) {
            return failCreate();
        }
        if (vkCreateFence(device, &fenceInfo, nullptr, &slot.computeFence) != VK_SUCCESS) {
            return failCreate();
        }

        slot.graphicsSubmittedThisLap = true;
        slot.computeSubmittedThisLap = true;
    }

    return true;
}

void FrameSync::shutdown() {
    if (device != VK_NULL_HANDLE) {
        for (FrameSlot& slot : frameSlots) {
            if (slot.renderFinishedSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, slot.renderFinishedSemaphore, nullptr);
            }
            if (slot.imageAvailableSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, slot.imageAvailableSemaphore, nullptr);
            }
            if (slot.computeFinishedSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, slot.computeFinishedSemaphore, nullptr);
            }
            if (slot.graphicsFence != VK_NULL_HANDLE) {
                vkDestroyFence(device, slot.graphicsFence, nullptr);
            }
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

void FrameSync::waitForSlot() {
    if (device == VK_NULL_HANDLE || currentFrame >= frameSlots.size()) {
        return;
    }

    const FrameSlot& slot = frameSlots[currentFrame];

    if (slot.graphicsSubmittedThisLap) {
        vkWaitForFences(device, 1, &slot.graphicsFence, VK_TRUE, UINT64_MAX);
    }

    if (slot.computeSubmittedThisLap) {
        vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, UINT64_MAX);
    }
}

void FrameSync::waitForAllFrameFences() {
    if (device == VK_NULL_HANDLE || frameSlots.empty()) {
        return;
    }

    for (const FrameSlot& slot : frameSlots) {
        vkWaitForFences(device, 1, &slot.graphicsFence, VK_TRUE, UINT64_MAX);
        vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, UINT64_MAX);
    }
}

VkResult FrameSync::acquireNextImage(VkSwapchainKHR swapChain, uint32_t& imageIndex) const {
    imageIndex = 0;
    if (device == VK_NULL_HANDLE ||
        swapChain == VK_NULL_HANDLE ||
        currentFrame >= frameSlots.size()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return vkAcquireNextImageKHR(
        device,
        swapChain,
        UINT64_MAX,
        frameSlots[currentFrame].imageAvailableSemaphore,
        VK_NULL_HANDLE,
        &imageIndex);
}

VkResult FrameSync::submitFrame(VkQueue computeQueue, VkQueue graphicsQueue,
                                VkSwapchainKHR swapChain, uint32_t imageIndex,
                                const FrameSubmission& submission) {
    (void)swapChain;
    (void)imageIndex;

    if (device == VK_NULL_HANDLE || currentFrame >= frameSlots.size()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    FrameSlot& slot = frameSlots[currentFrame];

    slot.graphicsSubmittedThisLap = false;
    slot.computeSubmittedThisLap = false;

    if (submission.computeCommandBuffer != VK_NULL_HANDLE) {
        vkResetFences(device, 1, &slot.computeFence);

        std::array<VkSemaphore, 1> waitSemaphores{};
        std::array<VkPipelineStageFlags, 1> waitStages{};
        std::array<uint64_t, 1> waitValues{};
        uint32_t waitCount = 0;
        if (submission.externalWaitSemaphore != VK_NULL_HANDLE) {
            waitSemaphores[waitCount] = submission.externalWaitSemaphore;
            waitStages[waitCount] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            waitValues[waitCount] = submission.externalWaitValue;
            ++waitCount;
        }

        std::array<VkSemaphore, 2> signalSemaphores{};
        std::array<uint64_t, 2> signalValues{};
        uint32_t signalCount = 0;
        if (submission.waitForComputeSemaphore) {
            signalSemaphores[signalCount] = slot.computeFinishedSemaphore;
            signalValues[signalCount] = 0;
            ++signalCount;
        }
        if (submission.externalSignalSemaphore != VK_NULL_HANDLE) {
            signalSemaphores[signalCount] = submission.externalSignalSemaphore;
            signalValues[signalCount] = submission.externalSignalValue;
            ++signalCount;
        }

        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.waitSemaphoreValueCount = waitCount;
        timelineInfo.pWaitSemaphoreValues = waitValues.data();
        timelineInfo.signalSemaphoreValueCount = signalCount;
        timelineInfo.pSignalSemaphoreValues = signalValues.data();

        VkSubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmitInfo.pNext = &timelineInfo;
        computeSubmitInfo.waitSemaphoreCount = waitCount;
        computeSubmitInfo.pWaitSemaphores = waitSemaphores.data();
        computeSubmitInfo.pWaitDstStageMask = waitStages.data();
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &submission.computeCommandBuffer;
        computeSubmitInfo.signalSemaphoreCount = signalCount;
        computeSubmitInfo.pSignalSemaphores = signalSemaphores.data();

        const VkResult computeResult = vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, slot.computeFence);
        if (computeResult == VK_SUCCESS) {
            slot.computeSubmittedThisLap = true;
        } else {
            std::cerr << "[SUBMIT] COMPUTE failed VkResult=" << static_cast<int>(computeResult)
                      << " frame=" << currentFrame << std::endl;
            return computeResult;
        }
    }

    // Graphics submit 
    if (submission.graphicsCommandBuffer != VK_NULL_HANDLE) {
        vkResetFences(device, 1, &slot.graphicsFence);

        std::array<VkSemaphore, 2> waitSemaphores{};
        std::array<VkPipelineStageFlags, 2> waitStages{};
        uint32_t waitCount = 1;
        waitSemaphores[0] = slot.imageAvailableSemaphore;
        waitStages[0] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        if (submission.waitForComputeSemaphore) {
            waitSemaphores[waitCount] = slot.computeFinishedSemaphore;
            waitStages[waitCount] = submission.computeWaitDstStageMask;
            ++waitCount;
        }

        VkSubmitInfo graphicsSubmitInfo{};
        graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        graphicsSubmitInfo.waitSemaphoreCount = waitCount;
        graphicsSubmitInfo.pWaitSemaphores = waitSemaphores.data();
        graphicsSubmitInfo.pWaitDstStageMask = waitStages.data();
        graphicsSubmitInfo.commandBufferCount = 1;
        graphicsSubmitInfo.pCommandBuffers = &submission.graphicsCommandBuffer;
        graphicsSubmitInfo.signalSemaphoreCount = 1;
        graphicsSubmitInfo.pSignalSemaphores = &slot.renderFinishedSemaphore;

        const VkResult graphicsResult = vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, slot.graphicsFence);
        if (graphicsResult != VK_SUCCESS) {
            std::cerr << "[SUBMIT] GRAPHICS failed VkResult=" << static_cast<int>(graphicsResult)
                      << " frame=" << currentFrame << std::endl;
            return graphicsResult;
        }
        slot.graphicsSubmittedThisLap = true;
    }

    return VK_SUCCESS;
}

VkResult FrameSync::present(VkQueue presentQueue, VkSwapchainKHR swapChain, uint32_t imageIndex) const {
    if (presentQueue == VK_NULL_HANDLE || swapChain == VK_NULL_HANDLE || currentFrame >= frameSlots.size()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &frameSlots[currentFrame].renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapChain;
    presentInfo.pImageIndices = &imageIndex;

    return vkQueuePresentKHR(presentQueue, &presentInfo);
}

void FrameSync::advanceFrame() {
    if (frameCount == 0) {
        return;
    }

    currentFrame = (currentFrame + 1) % frameCount;
}

void FrameSync::resetFrameIndex() {
    currentFrame = 0;

    for (FrameSlot& slot : frameSlots) {
        slot.graphicsSubmittedThisLap = true;
        slot.computeSubmittedThisLap = true;
    }
}
