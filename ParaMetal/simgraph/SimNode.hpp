#pragma once

#include "framegraph/ComputePass.hpp"

#include <cstdint>

struct SimTickContext {
    uint64_t tick = 0;
    uint32_t renderFrameSlot = 0;
    float deltaSeconds = 0.0f;
    bool playing = false;
    uint32_t currentFrame = 0;
    uint32_t resetEpoch = 0;
};

enum class SimNodeState : uint8_t {
    Inactive,
    Ready,
    Running,
    Paused,
    Completed,
};

class SimNode : public ComputePass {
public:
    void setTickContext(const SimTickContext& context) { tickContext = context; }
    const SimTickContext& getTickContext() const { return tickContext; }

    void update() final { update(tickContext); }
    virtual void update(const SimTickContext& context) = 0;

    bool hasDispatchableComputeWork() const final {
        return nodeState != SimNodeState::Inactive &&
            isSubmissionReady() &&
            hasPendingSubmission();
    }

    virtual bool isSubmissionReady() const = 0;
    virtual bool hasPendingSubmission() const = 0;

    SimNodeState getSimNodeState() const { return nodeState; }
    bool hasStartedSimulation() const { return hasStarted; }

    virtual bool hasTimelineMetadata() const { return false; }
    virtual float getSimulationDurationSeconds() const { return 0.0f; }
    virtual float getSimulationTimeSeconds() const { return 0.0f; }
    virtual uint32_t getTimelineFrameCount() const { return 0; }
    virtual uint32_t getRecordedTimelineFrames() const { return 0; }
    virtual uint32_t getPlaybackAvailableFrame() const { return UINT32_MAX; }

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

protected:
    void updateLifecycleState(bool inputsValid, bool completed, bool playing) {
        const SimNodeState previous = nodeState;
        if (!inputsValid) {
            hasStarted = false;
            nodeState = SimNodeState::Inactive;
        } else if (completed) {
            hasStarted = true;
            nodeState = SimNodeState::Completed;
        } else if (playing) {
            hasStarted = true;
            nodeState = SimNodeState::Running;
        } else {
            nodeState = hasStarted ? SimNodeState::Paused : SimNodeState::Ready;
        }

        if (nodeState == SimNodeState::Inactive && previous != SimNodeState::Inactive) {
            onEnteredInactive();
        }
    }

    void resetLifecycleState() {
        nodeState = SimNodeState::Inactive;
        hasStarted = false;
    }

    virtual void onEnteredInactive() {}

private:
    SimTickContext tickContext{};
    SimNodeState nodeState = SimNodeState::Inactive;
    bool hasStarted = false;
};
