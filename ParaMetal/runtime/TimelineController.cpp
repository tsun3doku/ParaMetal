#include "TimelineController.hpp"

#include "TimelineRuntime.hpp"

TimelineController::TimelineController(TimelineRuntime* runtime_)
    : runtime(runtime_) {
}

void TimelineController::bind(TimelineRuntime* runtime_) {
    runtime = runtime_;
}

void TimelineController::setPlaying(bool playing) {
    if (runtime) {
        runtime->setPlaying(playing);
    }
}

void TimelineController::togglePlaying() {
    if (runtime) {
        runtime->setPlaying(!runtime->isPlaying());
    }
}

void TimelineController::reset() {
    if (runtime) {
        runtime->reset();
    }
}

void TimelineController::scrubToFrame(uint32_t frame) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->setCurrentFrame(frame);
    }
}

void TimelineController::stepFrames(int deltaFrames) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->stepFrames(deltaFrames);
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
