#include "TimelineController.hpp"

#include "TimelineRuntime.hpp"
#include "TimelinePlaybackTarget.hpp"

TimelineController::TimelineController(TimelineRuntime* runtime_)
    : runtime(runtime_) {
}

void TimelineController::bindPlaybackTarget(TimelinePlaybackTarget* target) {
    playbackTarget = target;
}

void TimelineController::setPlaying(bool playing) {
    if (runtime) {
        if (playing && runtime->getCurrentFrame() >= runtime->getMaxFrame()) {
            runtime->reset();
            if (playbackTarget) playbackTarget->resetTimeline();
        }
        runtime->setPlaying(playing);
    }
    if (playbackTarget) playbackTarget->setTimelinePlaying(playing);
}

void TimelineController::reset() {
    if (runtime) {
        runtime->reset();
    }
    if (playbackTarget) playbackTarget->resetTimeline();
}

void TimelineController::scrubToFrame(uint32_t frame) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->setCurrentFrame(frame);
    }
    if (playbackTarget) playbackTarget->scrubTimeline(frame);
}

void TimelineController::stepFrames(int deltaFrames) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->stepFrames(deltaFrames);
        if (playbackTarget) playbackTarget->scrubTimeline(runtime->getCurrentFrame());
    }
}

void TimelineController::setFrameCount(uint32_t frameCount) {
    if (runtime) {
        runtime->setFrameCount(frameCount);
    }
}

void TimelineController::setFps(float fps) {
    if (runtime) {
        runtime->setFps(fps);
    }
}
