#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>
struct FrameComputeSubmission;

class FrameSync {
public:
    bool initialize(VkDevice device, uint32_t maxFramesInFlight);
    void shutdown();

    VkResult prepareExternalFrame(uint32_t frameIndex);
    bool wasComputePresentedThisLap() const;

    bool waitForAllFrameFences(uint64_t timeoutNs);

    VkResult submitComputeBatch(VkQueue computeQueue, const std::vector<FrameComputeSubmission>& submissions);

private:
    struct FrameSlot {
        VkFence computeFence = VK_NULL_HANDLE;
        bool computeSubmittedThisLap = false;
    };

    VkDevice device = VK_NULL_HANDLE;
    uint32_t frameCount = 0;
    uint32_t currentFrame = 0;
    std::vector<FrameSlot> frameSlots;
};
