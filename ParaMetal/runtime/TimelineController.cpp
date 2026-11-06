#include "TimelineController.hpp"

#include "TimelineRuntime.hpp"

TimelineController::TimelineController(TimelineRuntime* runtime_)
    : runtime(runtime_) {
}

void TimelineController::setPlaying(bool playing) {
    if (!runtime) {
        return;
    }
    if (playing && runtime->getCurrentFrame() >= runtime->getMaxFrame()) {
        runtime->reset();
    }
    runtime->setPlaying(playing);
}

void TimelineController::reset() {
    if (runtime) {
        runtime->reset();
    }
}

void TimelineController::scrubToFrame(uint32_t frame) {
    if (!runtime) {
        return;
    }
    runtime->setPlaying(false);
    runtime->setCurrentFrame(frame);
}

void TimelineController::stepFrames(int deltaFrames) {
    if (!runtime) {
        return;
    }
    runtime->setPlaying(false);
    runtime->stepFrames(deltaFrames);
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
