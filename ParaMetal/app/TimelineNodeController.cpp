#include "TimelineNodeController.hpp"

#include "App.h"
#include "TimelineWidget.hpp"
#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "runtime/RuntimeInterfaces.hpp"
#include "runtime/TimelineController.hpp"
#include "runtime/TimelineRuntime.hpp"

#include <QTimer>

#include <algorithm>
#include <cmath>
#include <cstdint>

TimelineNodeController::TimelineNodeController(QObject* parent)
    : QObject(parent) {
    syncTimer = new QTimer(this);
    syncTimer->setInterval(100);
    connect(syncTimer, &QTimer::timeout, this, &TimelineNodeController::syncAuthoredTimelineRange);
    syncTimer->start();
}

void TimelineNodeController::setApp(App* app_) {
    app = app_;
    lastTimelineRangeRevision = UINT64_MAX;
    updateTimelineBinding();
    syncAuthoredTimelineRange();
}

void TimelineNodeController::setTimelineWidget(TimelineWidget* widget) {
    if (timelineWidget == widget) {
        return;
    }

    if (timelineWidget) {
        disconnect(timelineWidget, nullptr, this, nullptr);
    }

    timelineWidget = widget;

    if (timelineWidget) {
        connect(timelineWidget, &TimelineWidget::playToggled, this, &TimelineNodeController::onPlayToggled, Qt::UniqueConnection);
        connect(timelineWidget, &TimelineWidget::resetClicked, this, &TimelineNodeController::onReset, Qt::UniqueConnection);
        connect(timelineWidget, &TimelineWidget::scrubbedToFrame, this, &TimelineNodeController::onScrub, Qt::UniqueConnection);
    }

    updateTimelineBinding();
}

void TimelineNodeController::updateTimelineBinding() {
    if (timelineWidget) {
        timelineWidget->bind(app ? app->runtimeQuery() : nullptr);
    }
}

void TimelineNodeController::syncAuthoredTimelineRange() {
    const RuntimeQuery* runtimeQuery = app ? app->runtimeQuery() : nullptr;
    if (runtimeQuery && runtimeQuery->isSimulationActive()) {
        return;
    }

    NodeGraphBridge* b = bridge();
    TimelineController* timeline = app ? app->timelineController() : nullptr;
    if (!b || !timeline) {
        return;
    }

    const uint64_t revision = b->getRevision();
    if (revision == lastTimelineRangeRevision) {
        return;
    }

    const NodeGraphNodeId nodeId = findTimelineHeatSolveNode();
    if (!nodeId.isValid()) {
        return;
    }

    NodeGraphNode node{};
    if (!b->getNode(nodeId, node)) {
        return;
    }

    const HeatSolveNodeParams params = readHeatSolveNodeParams(node);
    constexpr float authoredTimelineFps = TimelineRuntime::DefaultFps;
    const float durationSeconds = std::max(0.0f, static_cast<float>(params.simulationDuration));
    const uint32_t maxFrame = durationSeconds > 0.0f
        ? std::max(1u, static_cast<uint32_t>(std::ceil(durationSeconds * authoredTimelineFps)))
        : 0u;
    timeline->setFps(authoredTimelineFps);
    timeline->setFrameCount(maxFrame + 1u);
    lastTimelineRangeRevision = revision;
}

NodeGraphBridge* TimelineNodeController::bridge() const {
    return app ? app->getNodeGraphBridge() : nullptr;
}

NodeGraphNodeId TimelineNodeController::findTimelineHeatSolveNode() const {
    NodeGraphBridge* b = bridge();
    if (!b) {
        return {};
    }

    const NodeGraphState state = b->state();
    NodeGraphNodeId timelineNodeId{};
    for (const auto& [id, node] : state.nodes) {
        if (getNodeTypeId(node.typeId) == nodegraphtypes::HeatSolve) {
            if (timelineNodeId.isValid()) {
                return {};
            }
            timelineNodeId = node.id;
        }
    }

    return timelineNodeId;
}

bool TimelineNodeController::isTimelineAtEnd() const {
    const RuntimeQuery* runtimeQuery = app ? app->runtimeQuery() : nullptr;
    if (!runtimeQuery) {
        return false;
    }

    const uint32_t frameCount = runtimeQuery->getTimelineFrameCount();
    if (frameCount == 0) {
        return false;
    }

    return runtimeQuery->getTimelineCurrentFrame() >= frameCount - 1u;
}

void TimelineNodeController::onPlayToggled(bool playing) {
    const bool shouldRestartFromEnd = playing && isTimelineAtEnd();

    if (TimelineController* timeline = app ? app->timelineController() : nullptr) {
        if (shouldRestartFromEnd) {
            timeline->reset();
        }
        timeline->setPlaying(playing);
    }

    NodeGraphBridge* b = bridge();
    NodeGraphNodeId nodeId = findTimelineHeatSolveNode();
    if (!b || !nodeId.isValid()) {
        return;
    }

    if (!playing) {
        b->setNodeParameter(nodeId, NodeGraphParamValue{
            nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, true});
        return;
    }

    if (shouldRestartFromEnd) {
        NodeGraphNode node{};
        if (b->getNode(nodeId, node)) {
            const HeatSolveNodeParams params = readHeatSolveNodeParams(node);
            b->setNodeParameter(nodeId, NodeGraphParamValue{
                nodegraphparams::heatsolve::ResetRequested,
                NodeGraphParamType::Int,
                0.0,
                static_cast<int64_t>(params.resetCounter + 1),
                false});
        }
    }

    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::RewindFrame, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(heat::NoRewindFrame), false});
    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::Enabled, NodeGraphParamType::Bool, 0.0, 0, true});
    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, false});
}

void TimelineNodeController::onReset() {
    if (TimelineController* timeline = app ? app->timelineController() : nullptr) {
        timeline->reset();
    }

    NodeGraphBridge* b = bridge();
    NodeGraphNodeId nodeId = findTimelineHeatSolveNode();
    if (!b || !nodeId.isValid()) {
        return;
    }

    NodeGraphNode node{};
    if (!b->getNode(nodeId, node)) {
        return;
    }
    HeatSolveNodeParams params = readHeatSolveNodeParams(node);

    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, false});
    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::ResetRequested, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(params.resetCounter + 1), false});
    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::RewindFrame, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(heat::NoRewindFrame), false});
}

void TimelineNodeController::onScrub(uint32_t frame) {
    if (TimelineController* timeline = app ? app->timelineController() : nullptr) {
        timeline->scrubToFrame(frame);
    }

    NodeGraphBridge* b = bridge();
    NodeGraphNodeId nodeId = findTimelineHeatSolveNode();
    if (!b || !nodeId.isValid()) {
        return;
    }

    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::Paused, NodeGraphParamType::Bool, 0.0, 0, true});
    b->setNodeParameter(nodeId, NodeGraphParamValue{
        nodegraphparams::heatsolve::RewindFrame, NodeGraphParamType::Int, 0.0, static_cast<int64_t>(frame), false});
}
