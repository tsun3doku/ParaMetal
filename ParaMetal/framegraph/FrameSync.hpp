#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>
struct FrameComputeSubmission;

// Owns compute frame slots and coordinates compute-to-graphics sync transitions
class FrameSync {
public:
    bool initialize(VkDevice device, uint32_t maxFramesInFlight);
    void shutdown();

    void prepareExternalFrame(uint32_t frameIndex);
    // Whether a compute batch was submitted on this slot lap for graphics dependencies
    bool wasComputePresentedThisLap() const;

    void waitForAllFrameFences();

    VkResult submitComputeBatch(
        VkQueue computeQueue,
        const std::vector<FrameComputeSubmission>& submissions);

private:
    struct FrameSlot {
        VkFence computeFence = VK_NULL_HANDLE;
        bool computeSubmittedThisLap = false;
        bool computePresentedThisLap = false;
    };

    VkDevice device = VK_NULL_HANDLE;
    uint32_t frameCount = 0;
    uint32_t currentFrame = 0;
    std::vector<FrameSlot> frameSlots;
};
