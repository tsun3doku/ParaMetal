#include "TimelineRuntime.hpp"

#include <cmath>

void TimelineRuntime::reset() {
    currentFrame = 0;
    frameAccumulator = 0.0f;
    playing = false;
}

void TimelineRuntime::tick(float deltaSeconds) {
    if (!playing || fps <= 0.0f || frameCount == 0) {
        return;
    }

    frameAccumulator += std::max(0.0f, deltaSeconds) * fps;
    const uint32_t advanceFrames = static_cast<uint32_t>(std::floor(frameAccumulator));
    if (advanceFrames == 0) {
        return;
    }

    frameAccumulator -= static_cast<float>(advanceFrames);
    const uint32_t maxFrame = getMaxFrame();
    if (currentFrame + advanceFrames >= maxFrame) {
        currentFrame = maxFrame;
        playing = false;
        frameAccumulator = 0.0f;
        return;
    }

    currentFrame += advanceFrames;
}

void TimelineRuntime::setPlaying(bool playing_) {
    playing = playing_;
    if (!playing) {
        frameAccumulator = 0.0f;
    }
}

void TimelineRuntime::setCurrentFrame(uint32_t frame) {
    currentFrame = std::min(frame, getMaxFrame());
    frameAccumulator = 0.0f;
}

void TimelineRuntime::stepFrames(int deltaFrames) {
    const int64_t nextFrame = static_cast<int64_t>(currentFrame) + static_cast<int64_t>(deltaFrames);
    const int64_t clampedFrame = std::max<int64_t>(0, std::min<int64_t>(nextFrame, static_cast<int64_t>(getMaxFrame())));
    setCurrentFrame(static_cast<uint32_t>(clampedFrame));
}

void TimelineRuntime::setFrameCount(uint32_t frames) {
    frameCount = std::max(1u, frames);
    currentFrame = std::min(currentFrame, getMaxFrame());
}

void TimelineRuntime::setFps(float framesPerSecond) {
    fps = framesPerSecond > 0.0f ? framesPerSecond : DefaultFps;
}

float TimelineRuntime::getDurationSeconds() const {
    if (fps <= 0.0f || frameCount <= 1u) {
        return 0.0f;
    }
    return static_cast<float>(frameCount - 1u) / fps;
}

float TimelineRuntime::getCurrentSeconds() const {
    if (fps <= 0.0f) {
        return 0.0f;
    }
    return static_cast<float>(currentFrame) / fps;
}
