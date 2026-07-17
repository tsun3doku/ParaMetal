#include "ViewportPane.hpp"

#include "App.h"
#include "ViewportOverlayBar.hpp"
#include "VulkanWindow.hpp"

#include <QShowEvent>
#include <QSizePolicy>
#include <QVBoxLayout>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

ViewportPane::ViewportPane(QWidget* parent)
    : QWidget(parent) {
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    viewportWindow = new VulkanWindow();
    QWidget* viewportContainer = QWidget::createWindowContainer(viewportWindow, this);
    viewportContainer->setFocusPolicy(Qt::StrongFocus);
    viewportContainer->setMinimumSize(320, 240);
    layout->addWidget(viewportContainer);

    overlayBar = new ViewportOverlayBar(this);
}

void ViewportPane::setApp(App* app) {
    viewportWindow->setApp(app);
    overlayBar->bind(app ? app->getSettingsController() : nullptr);
}

void ViewportPane::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    initializeNativeLayering();
}

void ViewportPane::initializeNativeLayering() {
#ifdef Q_OS_WIN
    const HWND vulkanHwnd = reinterpret_cast<HWND>(viewportWindow->winId());
    if (vulkanHwnd) {
        const LONG_PTR exStyle = GetWindowLongPtr(vulkanHwnd, GWL_EXSTYLE);
        SetWindowLongPtr(vulkanHwnd, GWL_EXSTYLE, exStyle | WS_EX_NOREDIRECTIONBITMAP);
    }
#endif
    overlayBar->show();
}
