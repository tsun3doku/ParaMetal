#include "TimelineWidget.hpp"

#include "nodegraph/NodeGraphIconRegistry.hpp"
#include "runtime/RuntimeInterfaces.hpp"
#include "util/UiTheme.hpp"

#include <QIcon>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QResizeEvent>
#include <QTimer>
#include <QTransform>

#include <algorithm>
#include <cstdint>
#include <cmath>

static constexpr int TimelineWidgetHeight = 60;
static constexpr int TimelineMarginX = 18;
static constexpr int TimelineTrackHeight = 5;
static constexpr int TimelineButtonSize = 30;
static constexpr int TimelineButtonGap = 2;
static constexpr int TimelineFrameBoxWidth = 42;
static constexpr int TimelineFrameBoxHeight = 20;
static constexpr uint32_t TimelineDefaultEndFrame = 250;

QIcon TimelineWidget::loadPlaybackIcon(const QString& folder, bool mirrorHorizontal) {
    QPixmap pixmap(NodeGraphIconRegistry::iconPathForFolder(folder, 32.0));
    if (pixmap.isNull()) {
        return {};
    }

    if (mirrorHorizontal) {
        pixmap = pixmap.transformed(QTransform().scale(-1.0, 1.0), Qt::SmoothTransformation);
    }

    return QIcon(pixmap);
}

QPushButton* TimelineWidget::createTransportButton(const QIcon& icon, const QString& tooltip) {
    QPushButton* button = new QPushButton(this);
    button->setCursor(Qt::PointingHandCursor);
    button->setStyleSheet(QString::fromStdString(ui::toolButtonStyle()));
    button->setToolTip(tooltip);
    button->setIcon(icon);
    button->setIconSize(QSize(14, 14));
    return button;
}

TimelineWidget::TimelineWidget(QWidget* parent)
    : QWidget(parent) {
    setFixedHeight(TimelineWidgetHeight);
    setMouseTracking(true);

    playIcon = loadPlaybackIcon(QStringLiteral("Playback/Start"));
    pauseIcon = loadPlaybackIcon(QStringLiteral("Playback/Pause"));

    firstButton = createTransportButton(loadPlaybackIcon(QStringLiteral("Playback/Last_frame"), true), QStringLiteral("First frame"));
    connect(firstButton, &QPushButton::clicked, this, &TimelineWidget::onFirstClicked);

    previousButton = createTransportButton(loadPlaybackIcon(QStringLiteral("Playback/Next_frame"), true), QStringLiteral("Previous frame"));
    connect(previousButton, &QPushButton::clicked, this, &TimelineWidget::onPreviousClicked);

    playButton = new QPushButton(this);
    playButton->setCursor(Qt::PointingHandCursor);
    playButton->setStyleSheet(QString::fromStdString(ui::toolButtonStyle()));
    playButton->setIcon(playIcon);
    playButton->setIconSize(QSize(16, 16));
    playButton->setToolTip(QStringLiteral("Play / pause"));
    connect(playButton, &QPushButton::clicked, this, &TimelineWidget::onPlayClicked);

    nextButton = createTransportButton(loadPlaybackIcon(QStringLiteral("Playback/Next_frame")), QStringLiteral("Next frame"));
    connect(nextButton, &QPushButton::clicked, this, &TimelineWidget::onNextClicked);

    lastButton = createTransportButton(loadPlaybackIcon(QStringLiteral("Playback/Last_frame")), QStringLiteral("Last recorded frame"));
    connect(lastButton, &QPushButton::clicked, this, &TimelineWidget::onLastClicked);

    resetButton = new QPushButton(this);
    resetButton->setCursor(Qt::PointingHandCursor);
    resetButton->setStyleSheet(playButton->styleSheet());
    resetButton->setIcon(loadPlaybackIcon(QStringLiteral("Playback/Reset")));
    resetButton->setIconSize(QSize(14, 14));
    resetButton->setToolTip(QStringLiteral("Reset simulation"));
    connect(resetButton, &QPushButton::clicked, this, &TimelineWidget::resetClicked);

    layoutControls();

    pollTimer = new QTimer(this);
    pollTimer->setInterval(16); // ~60Hz
    connect(pollTimer, &QTimer::timeout, this, &TimelineWidget::updateState);
    pollTimer->start();
}

void TimelineWidget::bind(const RuntimeQuery* query) {
    runtimeQuery = query;
}

void TimelineWidget::updateState() {
    if (!runtimeQuery) {
        active = false;
        currentSeconds = 0.0f;
        durationSeconds = 0.0f;
        recordedFrames = 0;
        timelineFrameCount = 0;
        rewindFrame = UINT32_MAX;
        paused = false;

        firstButton->setEnabled(false);
        previousButton->setEnabled(false);
        playButton->setEnabled(false);
        playButton->setIcon(playIcon);
        nextButton->setEnabled(false);
        lastButton->setEnabled(false);
        resetButton->setEnabled(false);
        update();
        return;
    }

    active = runtimeQuery->isSimulationActive();
    paused = !runtimeQuery->isTimelinePlaying();
    currentSeconds = runtimeQuery->getTimelineCurrentSeconds();
    recordedFrames = runtimeQuery->getSimulationRecordedTimelineFrames();
    const uint32_t timelineFrames = runtimeQuery->getTimelineFrameCount();
    timelineFrameCount = timelineFrames > 0 ? timelineFrames - 1u : TimelineDefaultEndFrame;
    durationSeconds = runtimeQuery->getTimelineDurationSeconds();
    rewindFrame = runtimeQuery->getTimelineCurrentFrame();

    playButton->setEnabled(true);
    resetButton->setEnabled(true);
    firstButton->setEnabled(true);
    previousButton->setEnabled(true);
    nextButton->setEnabled(true);
    lastButton->setEnabled(true);

    updateShowPlayIcon();
    playButton->setIcon(showPlayIcon ? playIcon : pauseIcon);

    update();
}

void TimelineWidget::onPlayClicked() {
    emit playToggled(paused);
}

void TimelineWidget::onFirstClicked() {
    scrubToFrame(0);
}

void TimelineWidget::onPreviousClicked() {
    const uint32_t frame = currentDisplayFrame();
    scrubToFrame(frame > 0 ? frame - 1u : 0u);
}

void TimelineWidget::onNextClicked() {
    scrubToFrame(currentDisplayFrame() + 1u);
}

void TimelineWidget::onLastClicked() {
    scrubToFrame(maxTimelineFrame());
}

int TimelineWidget::trackLeft() const {
    return TimelineMarginX + TimelineButtonSize * 6 + TimelineButtonGap * 5 + 26;
}

int TimelineWidget::trackWidth() const {
    const int rightControls = 256;
    return std::max(1, width() - trackLeft() - rightControls - TimelineMarginX);
}

int TimelineWidget::trackCenterY() const {
    return 31;
}

uint32_t TimelineWidget::maxTimelineFrame() const {
    if (timelineFrameCount > 0) {
        return timelineFrameCount;
    }
        return TimelineDefaultEndFrame;
}

float TimelineWidget::secondsForFrame(uint32_t frame) const {
    const uint32_t maxFrame = maxTimelineFrame();
    if (durationSeconds <= 0.0f || maxFrame == 0) {
        return 0.0f;
    }
    frame = std::min(frame, maxFrame);
    return durationSeconds * (static_cast<float>(frame) / static_cast<float>(maxFrame));
}

uint32_t TimelineWidget::currentDisplayFrame() const {
    const uint32_t maxFrame = maxTimelineFrame();
    if (isDragging) {
        return std::min(dragStartFrame, maxFrame);
    }
    if (rewindFrame != UINT32_MAX) {
        return std::min(rewindFrame, maxFrame);
    }
    if (durationSeconds <= 0.0f || maxFrame == 0) {
        return 0;
    }
    return std::min(
        maxFrame,
        static_cast<uint32_t>(std::round(
            std::max(0.0f, std::min(durationSeconds, currentSeconds)) *
            static_cast<float>(maxFrame) / durationSeconds)));
}

uint32_t TimelineWidget::frameFromX(int x) const {
    const uint32_t maxFrame = maxTimelineFrame();
    if (maxFrame == 0) return 0;
    int left = trackLeft();
    int tw = trackWidth();
    float ratio = static_cast<float>(x - left) / static_cast<float>(tw);
    ratio = std::max(0.0f, std::min(1.0f, ratio));
    return static_cast<uint32_t>(std::round(ratio * static_cast<float>(maxFrame)));
}

void TimelineWidget::layoutControls() {
    const int y = (TimelineWidgetHeight - TimelineButtonSize) / 2;
    int x = TimelineMarginX;
    firstButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
    x += TimelineButtonSize + TimelineButtonGap;
    previousButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
    x += TimelineButtonSize + TimelineButtonGap;
    playButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
    x += TimelineButtonSize + TimelineButtonGap;
    nextButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
    x += TimelineButtonSize + TimelineButtonGap;
    lastButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
    x += TimelineButtonSize + TimelineButtonGap;
    resetButton->setGeometry(x, y, TimelineButtonSize, TimelineButtonSize);
}

void TimelineWidget::scrubToFrame(uint32_t frame) {
    const uint32_t clampedFrame = std::min(frame, maxTimelineFrame());
    dragStartFrame = clampedFrame;
    emit scrubbedToFrame(clampedFrame);
    update();
}

void TimelineWidget::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    p.fillRect(rect(), ui::TimelineBackground);

    int left = trackLeft();
    int tw = trackWidth();
    int centerY = trackCenterY();

    QRect trackRect(left, centerY - TimelineTrackHeight / 2, tw, TimelineTrackHeight);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(65, 67, 76));
    p.drawRect(trackRect);

    const uint32_t maxFrame = maxTimelineFrame();
    const uint32_t availableFrame = (recordedFrames > 0) ? std::min(recordedFrames - 1u, maxFrame) : 0u;
    const uint32_t frame = currentDisplayFrame();

    if (maxFrame > 0 && recordedFrames > 0) {
        const float availableRatio = static_cast<float>(availableFrame) / static_cast<float>(maxFrame);
        int fillWidth = static_cast<int>(std::round(static_cast<float>(tw) * availableRatio));
        fillWidth = std::max(1, std::min(tw, fillWidth));
        QRect fillRect(left, centerY - TimelineTrackHeight / 2, fillWidth, TimelineTrackHeight);
        p.setBrush(ui::TimelineTrackFill);
        p.drawRect(fillRect);
    }

    if (maxFrame > 0) {
        const uint32_t labelStep = std::max(1u, maxFrame / 10u);
        const uint32_t minorStep = std::max(1u, labelStep / 5u);
        const int tickTop = centerY - 6;
        auto drawTick = [&](uint32_t frameIndex, bool major) {
            const float ratio = static_cast<float>(frameIndex) / static_cast<float>(maxFrame);
            const int x = left + static_cast<int>(std::round(static_cast<float>(tw) * ratio));
            p.setPen(QPen(QColor(118, 119, 128), major ? 2 : 1));
            p.drawLine(x, tickTop - (major ? 1 : 0), x, tickTop + (major ? 13 : 11));
        };

        drawTick(0u, true);
        for (uint32_t frameIndex = minorStep; frameIndex <= maxFrame; frameIndex += minorStep) {
            const bool major = (frameIndex % labelStep) == 0u || frameIndex == maxFrame;
            drawTick(frameIndex, major);
            if (maxFrame - frameIndex < minorStep) {
                break;
            }
        }
        if ((maxFrame % minorStep) != 0u) {
            drawTick(maxFrame, true);
        }

        p.setFont(QFont("Segoe UI", 8));
        p.setPen(ui::TimelineText);
        for (uint32_t frameIndex = 0u; frameIndex <= maxFrame; frameIndex += labelStep) {
            const float ratio = static_cast<float>(frameIndex) / static_cast<float>(maxFrame);
            const int x = left + static_cast<int>(std::round(static_cast<float>(tw) * ratio));
            const QString label = QString::number(frameIndex);
            p.drawText(QRect(x - 28, centerY - 25, 56, 15), Qt::AlignHCenter | Qt::AlignVCenter, label);
            if (maxFrame - frameIndex < labelStep) {
                break;
            }
        }

        const int handleX = left + static_cast<int>(
            std::round(static_cast<float>(tw) * (static_cast<float>(frame) / static_cast<float>(maxFrame))));
        p.setPen(QPen(QColor(46, 126, 255), 2));
        p.drawLine(handleX, centerY - 9, handleX, centerY + 22);

        const QString bubbleText = QString::number(frame);
        const QFont bubbleFont("Segoe UI", 8, QFont::DemiBold);
        p.setFont(bubbleFont);
        const int bubbleWidth = std::max(32, p.fontMetrics().horizontalAdvance(bubbleText) + 12);
        QRect bubbleRect(handleX - bubbleWidth / 2, centerY - 29, bubbleWidth, 18);
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(53, 120, 255));
        p.drawRoundedRect(bubbleRect, 6, 6);
        p.setPen(QColor(245, 248, 255));
        p.drawText(bubbleRect, Qt::AlignCenter, bubbleText);
    }

    const int controlsRight = left + tw + 18;
    p.setFont(QFont("Segoe UI", 9));
    p.setPen(ui::TimelineText);
    const QRect startLabelRect(controlsRight, 8, 34, TimelineFrameBoxHeight);
    const QRect endLabelRect(controlsRight, 31, 34, TimelineFrameBoxHeight);
    p.drawText(startLabelRect, Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("Start"));
    p.drawText(endLabelRect, Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("End"));
    QRect startBox(controlsRight + 42, 8, TimelineFrameBoxWidth, TimelineFrameBoxHeight);
    QRect endBox(controlsRight + 42, 31, TimelineFrameBoxWidth, TimelineFrameBoxHeight);
    p.setPen(QPen(QColor(58, 60, 68), 1));
    p.setBrush(QColor(34, 35, 40));
    p.drawRoundedRect(startBox, 5, 5);
    p.drawRoundedRect(endBox, 5, 5);
    p.setPen(ui::TimelineText);
    p.drawText(startBox, Qt::AlignCenter, QStringLiteral("0"));
    p.drawText(endBox, Qt::AlignCenter, QString::number(maxFrame));

    p.setPen(ui::TimelineText);
    p.setFont(QFont("Segoe UI", 8));
    QString timeText = QString("%1 / %2 s")
        .arg(static_cast<double>(secondsForFrame(frame)), 0, 'f', 2)
        .arg(static_cast<double>(durationSeconds), 0, 'f', 2);
    const QRect timeRect(endBox.right() + 18, 20, std::max(1, width() - endBox.right() - 24), 18);
    p.drawText(timeRect, Qt::AlignLeft | Qt::AlignVCenter, timeText);
}

void TimelineWidget::resizeEvent(QResizeEvent* event) {
    layoutControls();
    QWidget::resizeEvent(event);
}

void TimelineWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }

    // Track scrub
    int left = trackLeft();
    int tw = trackWidth();
    if (event->pos().x() >= left && event->pos().x() <= left + tw) {
        isDragging = true;
        dragStartFrame = frameFromX(event->pos().x());
        scrubToFrame(dragStartFrame);
    }
}

void TimelineWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!isDragging) {
        return;
    }
    uint32_t frame = frameFromX(event->pos().x());
    scrubToFrame(frame);
}

void TimelineWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (isDragging) {
        isDragging = false;
        update();
    }
    (void)event;
}
