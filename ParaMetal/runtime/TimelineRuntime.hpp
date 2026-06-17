#pragma once

#include <algorithm>
#include <cstdint>

class TimelineRuntime {
public:
    static constexpr uint32_t DefaultFrameCount = 251;
    static constexpr float DefaultFps = 60.0f;

    void reset();
    void tick(float deltaSeconds);

    void setPlaying(bool playing);
    bool isPlaying() const { return playing; }

    void setCurrentFrame(uint32_t frame);
    uint32_t getCurrentFrame() const { return currentFrame; }

    void stepFrames(int deltaFrames);

    void setFrameCount(uint32_t frames);
    uint32_t getFrameCount() const { return frameCount; }
    uint32_t getMaxFrame() const { return frameCount > 0 ? frameCount - 1u : 0u; }

    uint32_t getStartDisplayFrame() const { return 0u; }
    uint32_t getEndDisplayFrame() const { return getMaxFrame(); }

    void setFps(float framesPerSecond);
    float getFps() const { return fps; }
    float getDurationSeconds() const;
    float getCurrentSeconds() const;

private:
    uint32_t frameCount = DefaultFrameCount;
    uint32_t currentFrame = 0;
    float fps = DefaultFps;
    float frameAccumulator = 0.0f;
    bool playing = false;
};
