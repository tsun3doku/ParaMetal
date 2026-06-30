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
        // Fences are created signaled so the very first waitForSlot (before any
        // submit has run) succeeds immediately. Mark the slot as submitted to
        // match that signaled state.
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

    // Graphics always submits, so its fence is always signaled this lap.
    if (slot.graphicsSubmittedThisLap) {
        vkWaitForFences(device, 1, &slot.graphicsFence, VK_TRUE, UINT64_MAX);
    }

    // Compute fence is only signaled when compute work was submitted this lap.
    // Waiting on it unconditionally was the permanent-hang source: a frame
    // whose compute submit was skipped (or failed before signaling) left the
    // fence unsignaled, and the next lap's wait blocked forever.
    if (slot.computeSubmittedThisLap) {
        vkWaitForFences(device, 1, &slot.computeFence, VK_TRUE, UINT64_MAX);
    }
}

void FrameSync::waitForAllFrameFences() {
    if (device == VK_NULL_HANDLE || frameSlots.empty()) {
        return;
    }

    // One-off path (resource uploads). Wait on every slot's fences directly.
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

    // Clear this lap's submitted flags before submitting. They get set true
    // only when the corresponding vkQueueSubmit succeeds. If a submit fails or
    // is skipped, the flag stays false and the next waitForSlot on this slot
    // skips the wait - no unsignaled-fence hang.
    slot.graphicsSubmittedThisLap = false;
    slot.computeSubmittedThisLap = false;

    // Compute submit (optional). Reset + submit are bound together here, so a
    // reset can never exist without a signal in the same atomic operation.
    if (submission.computeCommandBuffer != VK_NULL_HANDLE) {
        vkResetFences(device, 1, &slot.computeFence);

        VkSubmitInfo computeSubmitInfo{};
        computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &submission.computeCommandBuffer;

        if (submission.waitForComputeSemaphore) {
            computeSubmitInfo.signalSemaphoreCount = 1;
            computeSubmitInfo.pSignalSemaphores = &slot.computeFinishedSemaphore;
        }

        const VkResult computeResult = vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, slot.computeFence);
        if (computeResult == VK_SUCCESS) {
            slot.computeSubmittedThisLap = true;
        } else {
            std::cerr << "[SUBMIT] COMPUTE failed VkResult=" << static_cast<int>(computeResult)
                      << " frame=" << currentFrame << std::endl;
            // On compute failure we do NOT submit graphics: the graphics
            // command may depend on compute output and the compute fence is
            // unsignaled. Bail with the error so the caller recreates.
            return computeResult;
        }
    }

    // Graphics submit (required).
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

    // NOTE: do NOT clear the submittedThisLap flags here. They persist across
    // laps because they answer "is this slot's fence in-flight?" - and once a
    // submitFrame sets a flag true, that fence stays in-flight until the NEXT
    // submitFrame on the same slot drains it (via waitForSlot) and resets it.
    // Clearing here would make the next waitForSlot skip the wait, reusing
    // in-flight command buffers and hanging the GPU.

    currentFrame = (currentFrame + 1) % frameCount;
}

void FrameSync::resetFrameIndex() {
    currentFrame = 0;
    // Fences survive a reset; mark all slots submitted so the next waitForSlot
    // doesn't hang on a slot that has no in-flight work yet.
    for (FrameSlot& slot : frameSlots) {
        slot.graphicsSubmittedThisLap = true;
        slot.computeSubmittedThisLap = true;
    }
}
