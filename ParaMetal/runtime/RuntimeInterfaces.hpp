#pragma once

class RuntimeQuery {
public:
    virtual ~RuntimeQuery() = default;

    virtual bool isSimulationActive() const = 0;
    virtual bool isSimulationPaused() const = 0;
    virtual float getSimulationTotalTime() const = 0;
    virtual uint32_t getSimulationRecordedTimelineFrames() const = 0;
    virtual uint32_t getSimulationTimelineFrameCount() const = 0;
    virtual float getSimulationDuration() const = 0;
    virtual uint32_t getSimulationResetCounter() const = 0;
    virtual uint32_t getSimulationRewindFrame() const = 0;

    virtual bool isTimelinePlaying() const = 0;
    virtual uint32_t getTimelineCurrentFrame() const = 0;
    virtual uint32_t getTimelineFrameCount() const = 0;
    virtual uint32_t getTimelineStartDisplayFrame() const = 0;
    virtual uint32_t getTimelineEndDisplayFrame() const = 0;
    virtual float getTimelineCurrentSeconds() const = 0;
    virtual float getTimelineDurationSeconds() const = 0;
};
