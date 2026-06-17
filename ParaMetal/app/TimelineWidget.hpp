#pragma once

#include <QIcon>
#include <QWidget>

class QTimer;
class QPushButton;
class RuntimeQuery;
class QString;

class QMouseEvent;
class QResizeEvent;

class TimelineWidget : public QWidget {
    Q_OBJECT

public:
    explicit TimelineWidget(QWidget* parent = nullptr);

    void bind(const RuntimeQuery* query);

signals:
    void playToggled(bool playing);
    void resetClicked();
    void scrubbedToFrame(uint32_t frame);

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private slots:
    void updateState();
    void onPlayClicked();
    void onFirstClicked();
    void onPreviousClicked();
    void onNextClicked();
    void onLastClicked();

private:
    uint32_t frameFromX(int x) const;
    uint32_t currentDisplayFrame() const;
    uint32_t maxTimelineFrame() const;
    float secondsForFrame(uint32_t frame) const;
    int trackLeft() const;
    int trackWidth() const;
    int trackCenterY() const;
    void layoutControls();
    void scrubToFrame(uint32_t frame);
    static QIcon loadPlaybackIcon(const QString& folder, bool mirrorHorizontal = false);
    QPushButton* createTransportButton(const QIcon& icon, const QString& tooltip);

    const RuntimeQuery* runtimeQuery = nullptr;

    QPushButton* firstButton = nullptr;
    QPushButton* previousButton = nullptr;
    QPushButton* playButton = nullptr;
    QPushButton* nextButton = nullptr;
    QPushButton* lastButton = nullptr;
    QPushButton* resetButton = nullptr;

    QIcon playIcon;
    QIcon pauseIcon;

    float currentSeconds = 0.0f;
    float durationSeconds = 0.0f;
    uint32_t recordedFrames = 0;
    uint32_t timelineFrameCount = 0;
    uint32_t rewindFrame = UINT32_MAX;
    bool paused = false;
    bool active = false;
    bool showPlayIcon = true;

    bool isDragging = false;
    uint32_t dragStartFrame = 0;

    QTimer* pollTimer = nullptr;

    void updateShowPlayIcon() {
        showPlayIcon = paused || isAtEnd();
    }

    bool isAtEnd() const {
        return timelineFrameCount > 0 && currentDisplayFrame() >= timelineFrameCount;
    }
};
