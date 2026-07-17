#pragma once

#include <QWidget>

class App;
class QShowEvent;
class ViewportOverlayBar;
class VulkanWindow;

class ViewportPane : public QWidget {
public:
    explicit ViewportPane(QWidget* parent = nullptr);

    void setApp(App* app);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void initializeNativeLayering();

    VulkanWindow* viewportWindow = nullptr;
    ViewportOverlayBar* overlayBar = nullptr;
};
