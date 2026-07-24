#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class ComputePass {
public:
    struct Synchronization {
        VkSemaphore waitSemaphore = VK_NULL_HANDLE;
        uint64_t waitValue = 0;
        VkSemaphore signalSemaphore = VK_NULL_HANDLE;
        uint64_t signalValue = 0;
    };

    virtual ~ComputePass() = default;

    virtual void update() = 0;
    virtual bool hasDispatchableComputeWork() const = 0;
    virtual const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const = 0;
    virtual void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) = 0;
    virtual void setComputeTimingQueries(VkQueryPool queryPool, uint32_t startQuery, uint32_t endQuery) {
        (void)queryPool;
        (void)startQuery;
        (void)endQuery;
    }
    virtual Synchronization getSynchronization() const { return {}; }
};
