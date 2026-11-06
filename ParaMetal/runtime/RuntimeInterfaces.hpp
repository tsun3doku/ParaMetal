#pragma once

#include "serial/SerialTemperatureRuntime.hpp"

#include <cstdint>
#include <vector>

class RuntimeQuery {
public:
    virtual ~RuntimeQuery() = default;

    virtual bool isSimulationActive() const = 0;
    virtual bool isSimulationPaused() const = 0;
    virtual uint32_t getSimulationRecordedTimelineFrames() const = 0;

    virtual bool isTimelinePlaying() const = 0;
    virtual uint32_t getTimelineCurrentFrame() const = 0;
    virtual uint32_t getTimelineFrameCount() const = 0;
    virtual uint32_t getTimelineStartDisplayFrame() const = 0;
    virtual uint32_t getTimelineEndDisplayFrame() const = 0;
    virtual float getTimelineCurrentSeconds() const = 0;
    virtual float getTimelineDurationSeconds() const = 0;
    virtual bool getSerialTemperatureStatus(uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const = 0;
    virtual bool getHeatProbeTemperatures(uint64_t sourceKey, std::vector<float>& outTemperaturesC) const = 0;
};
