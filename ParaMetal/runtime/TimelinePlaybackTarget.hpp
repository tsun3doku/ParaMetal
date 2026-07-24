#pragma once

#include <cstdint>

class TimelinePlaybackTarget {
public:
    virtual ~TimelinePlaybackTarget() = default;

    virtual void setTimelinePlaying(bool playing) = 0;
    virtual void resetTimeline() = 0;
    virtual void scrubTimeline(uint32_t frame) = 0;
};
