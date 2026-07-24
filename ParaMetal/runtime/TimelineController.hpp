#pragma once

#include <cstdint>

class TimelineRuntime;
class TimelinePlaybackTarget;

class TimelineController {
public:
    explicit TimelineController(TimelineRuntime* runtime = nullptr);

    void bindPlaybackTarget(TimelinePlaybackTarget* playbackTarget);

    void setPlaying(bool playing);
    void reset();
    void scrubToFrame(uint32_t frame);
    void stepFrames(int deltaFrames);
    void setFrameCount(uint32_t frameCount);
    void setFps(float fps);

private:
    TimelineRuntime* runtime = nullptr;
    TimelinePlaybackTarget* playbackTarget = nullptr;
};
