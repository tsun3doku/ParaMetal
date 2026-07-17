#pragma once

#include <QIcon>
#include <QObject>
#include <QPointer>

class RenderSettingsController;
class QPushButton;
class QWidget;

class ViewportOverlayBar : public QObject {
public:
    explicit ViewportOverlayBar(QWidget* viewportHost);

    void bind(RenderSettingsController* controller);
    void show();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void raiseNative();
    void reposition();
    void updateState();

    QWidget* viewportHost = nullptr;
    QWidget* wireframeGroupFrame = nullptr;
    QPointer<RenderSettingsController> controller;
    QPushButton* wireframeButton = nullptr;
    QPushButton* shadedButton = nullptr;
    QPushButton* gridButton = nullptr;
    QIcon wireframeIcon;
    QIcon wireframeSelectedIcon;
    QIcon shadedIcon;
    QIcon shadedSelectedIcon;
    QIcon gridIcon;
    QIcon gridSelectedIcon;
};
