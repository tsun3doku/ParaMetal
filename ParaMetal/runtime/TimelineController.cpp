#include "TimelineController.hpp"

#include "TimelineRuntime.hpp"
#include "TimelinePlaybackTarget.hpp"

#include <algorithm>

TimelineController::TimelineController(TimelineRuntime* runtime_)
    : runtime(runtime_) {
}

void TimelineController::bindPlaybackTarget(TimelinePlaybackTarget* target) {
    playbackTargets.clear();
    if (target) {
        playbackTargets.push_back(target);
    }
}

void TimelineController::addPlaybackTarget(TimelinePlaybackTarget* target) {
    if (target) {
        auto it = std::find(playbackTargets.begin(), playbackTargets.end(), target);
        if (it == playbackTargets.end()) {
            playbackTargets.push_back(target);
        }
    }
}

void TimelineController::removePlaybackTarget(TimelinePlaybackTarget* target) {
    auto it = std::find(playbackTargets.begin(), playbackTargets.end(), target);
    if (it != playbackTargets.end()) {
        playbackTargets.erase(it);
    }
}

void TimelineController::clearPlaybackTargets() {
    playbackTargets.clear();
}

void TimelineController::setPlaying(bool playing) {
    if (runtime) {
        if (playing && runtime->getCurrentFrame() >= runtime->getMaxFrame()) {
            runtime->reset();
            for (auto* target : playbackTargets) {
                if (target) target->resetTimeline();
            }
        }
        runtime->setPlaying(playing);
    }
    for (auto* target : playbackTargets) {
        if (target) target->setTimelinePlaying(playing);
    }
}

void TimelineController::reset() {
    if (runtime) {
        runtime->reset();
    }
    for (auto* target : playbackTargets) {
        if (target) target->resetTimeline();
    }
}

void TimelineController::scrubToFrame(uint32_t frame) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->setCurrentFrame(frame);
    }
    for (auto* target : playbackTargets) {
        if (target) target->scrubTimeline(frame);
    }
}

void TimelineController::stepFrames(int deltaFrames) {
    if (runtime) {
        runtime->setPlaying(false);
        runtime->stepFrames(deltaFrames);
        for (auto* target : playbackTargets) {
            if (target) target->scrubTimeline(runtime->getCurrentFrame());
        }
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
