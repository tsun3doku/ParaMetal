#pragma once

#include "framegraph/ComputePass.hpp"

#include <cstdint>

class SimNode : public ComputePass {
public:
    virtual VkSemaphore publicationReadySemaphore() const { return VK_NULL_HANDLE; }
    virtual uint64_t publicationReadyValue() const { return 0; }
    virtual VkSemaphore publicationReleaseSemaphore() const { return VK_NULL_HANDLE; }
    virtual uint64_t peekPublicationRelease() const { return 0; }
    virtual void commitPublicationRelease(uint64_t value) { (void)value; }
    virtual void setPublicationReleaseWait(uint64_t value) { (void)value; }
    virtual void bindConsumedInput(uint64_t connectionId, VkSemaphore producerReady, uint64_t producerReadyValue) {
        (void)connectionId;
        (void)producerReady;
        (void)producerReadyValue;
    }
    virtual void clearConsumedInputs() {}
    virtual bool consumedInput(uint64_t connectionId) const {
        (void)connectionId;
        return false;
    }

    virtual float computeGpuMs() const { return 0.0f; }
    virtual uint64_t completedStepCount() const { return 0; }
};
