#pragma once

#include <cstdint>
#include <vector>

class TimelineRuntime;
class TimelinePlaybackTarget;

class TimelineController {
public:
    explicit TimelineController(TimelineRuntime* runtime = nullptr);

    void bindPlaybackTarget(TimelinePlaybackTarget* playbackTarget);
    void addPlaybackTarget(TimelinePlaybackTarget* playbackTarget);
    void removePlaybackTarget(TimelinePlaybackTarget* playbackTarget);
    void clearPlaybackTargets();

    void setPlaying(bool playing);
    void reset();
    void scrubToFrame(uint32_t frame);
    void stepFrames(int deltaFrames);
    void setFrameCount(uint32_t frameCount);
    void setFps(float fps);

private:
    TimelineRuntime* runtime = nullptr;
    std::vector<TimelinePlaybackTarget*> playbackTargets;
};
