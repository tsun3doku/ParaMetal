#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class ComputePass {
public:
    struct TimelineWait {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t value = 0;
        VkPipelineStageFlags stageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    };

    struct TimelineSignal {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t value = 0;
    };

#pragma push_macro("signals")
#undef signals

    struct Synchronization {
        std::vector<TimelineWait> waits;
        std::vector<TimelineSignal> signals;

        void addWait(VkSemaphore semaphore, uint64_t value, VkPipelineStageFlags stageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) {
            if (semaphore != VK_NULL_HANDLE) {
                waits.push_back({ semaphore, value, stageMask });
            }
        }

        void addSignal(VkSemaphore semaphore, uint64_t value) {
            if (semaphore != VK_NULL_HANDLE) {
                signals.push_back({ semaphore, value });
            }
        }
    };

#pragma pop_macro("signals")

    virtual ~ComputePass() = default;

    virtual void update() = 0;
    virtual bool hasDispatchableComputeWork() const = 0;
    virtual const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const = 0;
    virtual void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) = 0;
    virtual Synchronization getSynchronization() const { return {}; }
};

struct FrameComputeSubmission {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    ComputePass::Synchronization synchronization{};
};
