#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class FrameSync {
public:
    struct FrameSubmission {
        VkCommandBuffer computeCommandBuffer = VK_NULL_HANDLE;   // optional
        VkCommandBuffer graphicsCommandBuffer = VK_NULL_HANDLE;  // required
        bool waitForComputeSemaphore = false;
        bool insertComputeToGraphicsBarrier = false;
        VkPipelineStageFlags computeWaitDstStageMask = 0;
    };

    bool initialize(VkDevice device, uint32_t maxFramesInFlight);
    void shutdown();

    void waitForSlot();

    void waitForAllFrameFences();

    VkResult acquireNextImage(VkSwapchainKHR swapChain, uint32_t& imageIndex) const;
    VkResult present(VkQueue presentQueue, VkSwapchainKHR swapChain, uint32_t imageIndex) const;
    VkResult submitFrame(VkQueue computeQueue, VkQueue graphicsQueue,
        VkSwapchainKHR swapChain, uint32_t imageIndex,
        const FrameSubmission& submission);

    void advanceFrame();
    void resetFrameIndex();
    uint32_t getCurrentFrameIndex() const { return currentFrame; }

private:
    struct FrameSlot {
        VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
        VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
        VkSemaphore computeFinishedSemaphore = VK_NULL_HANDLE;
        VkFence graphicsFence = VK_NULL_HANDLE;
        VkFence computeFence = VK_NULL_HANDLE;

        bool graphicsSubmittedThisLap = false;
        bool computeSubmittedThisLap = false;
    };

    VkDevice device = VK_NULL_HANDLE;
    uint32_t frameCount = 0;
    uint32_t currentFrame = 0;
    std::vector<FrameSlot> frameSlots;
};