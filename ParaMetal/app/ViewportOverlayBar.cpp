#include "ViewportOverlayBar.hpp"

#include "render/RenderSettings.hpp"
#include "runtime/RenderSettingsController.hpp"
#include "ui/UiIconRegistry.hpp"
#include "ui/UiTheme.hpp"

#include <QEvent>
#include <QPainterPath>
#include <QPushButton>
#include <QRegion>
#include <QSize>
#include <QString>
#include <QWidget>

#include <array>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static constexpr int overlayIconSize = 18;
static constexpr int overlayButtonSize = 32;
static constexpr int overlayButtonGap = 1;
static constexpr int overlayGroupGap = 4;
static constexpr int overlayGroupBorderWidth = 1;
static constexpr int overlayTopMargin = 10;
static constexpr int overlayCornerRadius = 4;
static constexpr int wireframeGroupWidth = overlayButtonSize * 2 + overlayButtonGap + overlayGroupBorderWidth * 2;
static constexpr int wireframeGroupHeight = overlayButtonSize + overlayGroupBorderWidth * 2;
static constexpr int standaloneGroupSize = overlayButtonSize + overlayGroupBorderWidth * 2;
static constexpr int overlayWidth = wireframeGroupWidth + overlayGroupGap + standaloneGroupSize;

static QWidget* makeGroupFrame(QWidget* parent, int width, int height) {
    QWidget* frame = new QWidget(parent);
    frame->setAttribute(Qt::WA_DontCreateNativeAncestors);
    frame->setAttribute(Qt::WA_NativeWindow);
    frame->setStyleSheet(QStringLiteral("background-color: %1;")
                             .arg(ui::ToolButtonBorder.name()));
    frame->setFixedSize(width, height);

    const QRect frameRect(0, 0, width, height);
    QPainterPath outerPath;
    outerPath.addRoundedRect(frameRect, overlayCornerRadius + overlayGroupBorderWidth, overlayCornerRadius + overlayGroupBorderWidth);

    const QRect innerRect = frameRect.adjusted(overlayGroupBorderWidth, overlayGroupBorderWidth, -overlayGroupBorderWidth, -overlayGroupBorderWidth);
    QPainterPath innerPath;
    innerPath.addRoundedRect(innerRect, overlayCornerRadius, overlayCornerRadius);

    const QRegion outerRegion(outerPath.toFillPolygon().toPolygon());
    const QRegion innerRegion(innerPath.toFillPolygon().toPolygon());
    frame->setMask(outerRegion.subtracted(innerRegion));
    return frame;
}

static QPushButton* makeSegmentButton(QWidget* parent, const QIcon& icon, ui::ToolButtonSegment segment) {
    QPushButton* button = new QPushButton(parent);
    button->setAttribute(Qt::WA_DontCreateNativeAncestors);
    button->setAttribute(Qt::WA_NativeWindow);
    button->setCursor(Qt::PointingHandCursor);
    button->setFocusPolicy(Qt::NoFocus);
    button->setCheckable(true);
    button->setStyleSheet(QString::fromStdString(ui::toolButtonStyle(ui::ToolButtonSegment::Middle)));
    button->setIcon(icon);
    button->setIconSize(QSize(overlayIconSize, overlayIconSize));
    button->setFixedSize(overlayButtonSize, overlayButtonSize);

    const QRect buttonRect(0, 0, overlayButtonSize, overlayButtonSize);
    QPainterPath maskPath;
    maskPath.addRoundedRect(buttonRect, overlayCornerRadius, overlayCornerRadius);
    QRegion buttonMask(maskPath.toFillPolygon().toPolygon());
    if (segment == ui::ToolButtonSegment::Leading) {
        buttonMask = buttonMask.united(QRegion(buttonRect.adjusted(overlayCornerRadius, 0, 0, 0)));
    } else if (segment == ui::ToolButtonSegment::Trailing) {
        buttonMask = buttonMask.united(QRegion(buttonRect.adjusted(0, 0, -overlayCornerRadius, 0)));
    }
    button->setMask(buttonMask);
    return button;
}

ViewportOverlayBar::ViewportOverlayBar(QWidget* host)
    : QObject(host), viewportHost(host) {

    wireframeIcon = QIcon(ui::IconRegistry::screenSpacePixmapForFolder(QStringLiteral("Overlays/wireframe"), overlayIconSize));
    shadedIcon = QIcon(ui::IconRegistry::screenSpacePixmapForFolder(QStringLiteral("Overlays/wireframe_shaded"), overlayIconSize));
    gridIcon = QIcon(ui::IconRegistry::screenSpacePixmapForFolder(QStringLiteral("Overlays/grid"), overlayIconSize));

    wireframeGroupFrame = makeGroupFrame(viewportHost, wireframeGroupWidth, wireframeGroupHeight);
    gridGroupFrame = makeGroupFrame(viewportHost, standaloneGroupSize, standaloneGroupSize);
    wireframeButton = makeSegmentButton(viewportHost, wireframeIcon, ui::ToolButtonSegment::Leading);
    shadedButton = makeSegmentButton(viewportHost, shadedIcon, ui::ToolButtonSegment::Trailing);
    gridButton = makeSegmentButton(viewportHost, gridIcon, ui::ToolButtonSegment::Standalone);
    wireframeButton->setAccessibleName(QStringLiteral("Wireframe view"));
    shadedButton->setAccessibleName(QStringLiteral("Shaded wireframe view"));
    gridButton->setAccessibleName(QStringLiteral("Viewport grid"));

    connect(wireframeButton, &QPushButton::clicked, this, [this] {
        if (!controller) {
            return;
        }
        const app::WireframeMode mode = controller->getSnapshot().wireframeMode;
        controller->setWireframeMode(mode == app::WireframeMode::Wireframe ? app::WireframeMode::Off : app::WireframeMode::Wireframe);
    });
    connect(shadedButton, &QPushButton::clicked, this, [this] {
        if (!controller) {
            return;
        }
        const app::WireframeMode mode = controller->getSnapshot().wireframeMode;
        controller->setWireframeMode(mode == app::WireframeMode::Shaded ? app::WireframeMode::Off : app::WireframeMode::Shaded);
    });
    connect(gridButton, &QPushButton::clicked, this, [this] {
        if (controller) {
            controller->toggleGrid();
        }
    });

    if (viewportHost) {
        viewportHost->installEventFilter(this);
    }
}

void ViewportOverlayBar::bind(RenderSettingsController* newController) {
    if (controller) {
        disconnect(controller, nullptr, this, nullptr);
    }
    controller = newController;
    if (controller) {
        connect(controller, &RenderSettingsController::settingsChanged, this, &ViewportOverlayBar::updateState);
        connect(controller, &QObject::destroyed, this, [this] { updateState(); });
    }
    updateState();
}

void ViewportOverlayBar::show() {
    reposition();
    wireframeGroupFrame->show();
    gridGroupFrame->show();
    wireframeButton->show();
    shadedButton->show();
    gridButton->show();
    raiseNative();
}

void ViewportOverlayBar::raiseNative() {
#ifdef Q_OS_WIN
    const std::array<QWidget*, 5> overlayWidgets = {
        wireframeGroupFrame, gridGroupFrame, wireframeButton, shadedButton, gridButton};
    for (QWidget* widget : overlayWidgets) {
        const WId wid = widget->winId();
        if (wid != 0) {SetWindowPos(reinterpret_cast<HWND>(wid), HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
        }
    }
#else
    wireframeGroupFrame->raise();
    gridGroupFrame->raise();
    wireframeButton->raise();
    shadedButton->raise();
    gridButton->raise();
#endif
}

bool ViewportOverlayBar::eventFilter(QObject* watched, QEvent* event) {
    if (watched == viewportHost && event->type() == QEvent::Resize) {
        reposition();
    }
    return QObject::eventFilter(watched, event);
}

void ViewportOverlayBar::reposition() {
    if (!viewportHost) {
        return;
    }
    const int left = (viewportHost->width() - overlayWidth) / 2;
    const int gridFrameLeft = left + wireframeGroupWidth + overlayGroupGap;
    wireframeGroupFrame->move(left, overlayTopMargin - overlayGroupBorderWidth);
    gridGroupFrame->move(gridFrameLeft, overlayTopMargin - overlayGroupBorderWidth);
    wireframeButton->move(left + overlayGroupBorderWidth, overlayTopMargin);
    shadedButton->move(left + overlayGroupBorderWidth + overlayButtonSize + overlayButtonGap, overlayTopMargin);
    gridButton->move(gridFrameLeft + overlayGroupBorderWidth, overlayTopMargin);
}

void ViewportOverlayBar::updateState() {
    wireframeButton->setEnabled(controller != nullptr);
    shadedButton->setEnabled(controller != nullptr);
    gridButton->setEnabled(controller != nullptr);
    if (!controller) {
        wireframeButton->setChecked(false);
        shadedButton->setChecked(false);
        gridButton->setChecked(false);
        return;
    }
    const app::RenderSettings settings = controller->getSnapshot();
    const app::WireframeMode mode = settings.wireframeMode;
    wireframeButton->setChecked(mode == app::WireframeMode::Wireframe);
    shadedButton->setChecked(mode == app::WireframeMode::Shaded);
    gridButton->setChecked(settings.gridEnabled);
}
