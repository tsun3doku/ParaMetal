#pragma once

#include "heat/HeatGpuStructs.hpp"
#include "nodegraph/NodeGraphTypes.hpp"

#include <QObject>
#include <QPointer>

#include <cstdint>

class App;
class NodeGraphBridge;
class QTimer;
class TimelineWidget;

class TimelineNodeController : public QObject {
    Q_OBJECT
public:
    explicit TimelineNodeController(QObject* parent = nullptr);

    void setApp(App* app);
    void setTimelineWidget(TimelineWidget* widget);

private slots:
    void syncAuthoredTimelineRange();
    void onPlayToggled(bool playing);
    void onReset();
    void onScrub(uint32_t frame);

private:
    NodeGraphNodeId findTimelineHeatSolveNode() const;
    bool isTimelineAtEnd() const;
    NodeGraphBridge* bridge() const;
    void updateTimelineBinding();

    App* app = nullptr;
    QPointer<TimelineWidget> timelineWidget;
    QTimer* syncTimer = nullptr;
    uint64_t lastTimelineRangeRevision = UINT64_MAX;
};
