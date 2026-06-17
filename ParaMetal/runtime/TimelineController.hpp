#pragma once

#include <cstdint>

class TimelineRuntime;

class TimelineController {
public:
    explicit TimelineController(TimelineRuntime* runtime = nullptr);

    void bind(TimelineRuntime* runtime);

    void setPlaying(bool playing);
    void togglePlaying();
    void reset();
    void scrubToFrame(uint32_t frame);
    void stepFrames(int deltaFrames);
    void setFrameCount(uint32_t frameCount);
    void setFps(float fps);

private:
    TimelineRuntime* runtime = nullptr;
};
