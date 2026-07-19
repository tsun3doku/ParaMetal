#pragma once

#include "ui/UiTheme.hpp"

#include <QColor>

class QString;
class QLineEdit;
class QLabel;
class QWidget;
class QComboBox;
class QFrame;

namespace nodegraphwidgets {

constexpr int panelMarginLeft = 22;
constexpr int panelMarginTop = 20;
constexpr int panelMarginRight = 22;
constexpr int panelMarginBottom = 20;
constexpr int panelContentMarginLeft = 12;
constexpr int panelContentMarginTop = 14;
constexpr int panelContentMarginRight = 12;
constexpr int panelContentMarginBottom = 14;
constexpr int panelContentSpacing = 12;
constexpr int panelCardInnerSpacing = 16;
constexpr int panelCardRadius = 10;
constexpr int panelWidgetRadius = 4;
constexpr int panelCheckboxIndicatorRadius = 4;
constexpr int panelCheckboxIndicatorSize = 14;
constexpr int panelCheckboxSpacing = 8;
constexpr int panelStatusTopPadding = 4;
constexpr int panelTabHorizontalPadding = 10;
constexpr int panelTabTopPadding = 8;
constexpr int panelTabBottomPadding = 10;
constexpr int panelTabRightMargin = 18;
constexpr int panelTabSelectedBorderWidth = 3;
constexpr int panelFieldVerticalPadding = 6;
constexpr int panelFieldHorizontalPadding = 8;
constexpr int panelButtonVerticalPadding = 7;
constexpr int panelButtonHorizontalPadding = 12;
constexpr int panelTitleBarHeight = 32;
constexpr int panelMinimumHeight = 260;
constexpr int panelDataflowHeight = 140;
constexpr int panelTableMinimumHeight = 120;
constexpr int heatStatusTimerIntervalMs = 125;

constexpr int actionStripHeight = 144;
constexpr int actionStripBannerRadius = 6;
constexpr int actionStripPreviewSize = 128;
constexpr int actionStripDismissSize = 24;
constexpr int actionStripDismissIconSize = 12;

constexpr int navHintIconSize = 28;
constexpr int navHintIconTextGap = 6;
constexpr int navHintItemSpacing = 18;
constexpr int navHintCanvasMargin = 12;

constexpr int sliderRowSpacing = 10;
constexpr int sliderLabelWidth = 120;
constexpr int sliderRightPadding = 10;
constexpr int sliderValueFieldWidth = 72;
constexpr int sliderTrackHeight = 6;
constexpr int sliderTrackRadius = 3;
constexpr int sliderKnobDiameter = 14;
constexpr int sliderKnobBorderWidth = 2;
constexpr int sliderKnobMaskPadding = 3;
constexpr int sliderHorizontalInset = 6;
constexpr int sliderMinimumHeight = 28;

inline constexpr QColor colorPanelBackground = ui::PanelBackground;
inline constexpr QColor colorPanelCardBackground = QColor(45, 44, 50);
inline constexpr QColor colorPanelCardBorder = QColor(58, 57, 60);
inline constexpr QColor colorTextPrimary = QColor(236, 234, 246);
inline constexpr QColor colorTextInput = QColor(243, 241, 251);
inline constexpr QColor colorTextHeading = QColor(246, 245, 251);
inline constexpr QColor colorTextSecondary = QColor(210, 209, 222);
inline constexpr QColor colorTextMuted = QColor(200, 198, 212);
inline constexpr QColor colorTextHover = QColor(229, 227, 239);
inline constexpr QColor colorTextTabSelected = QColor(244, 242, 255);
inline constexpr QColor colorStatusAccent = QColor(167, 183, 255);
inline constexpr QColor colorAccent = QColor(94, 124, 255);
inline constexpr QColor colorAccentFocus = QColor(126, 151, 255);
inline constexpr QColor colorAccentPressed = QColor(75, 103, 224);
inline constexpr QColor colorActionStripButtonHover = QColor(65, 128, 255);
inline constexpr QColor colorTextOnAccent = QColor(255, 255, 255);
inline constexpr QColor colorCheckboxBorder = QColor(119, 117, 139);
inline constexpr QColor colorCheckboxBackground = QColor(58, 57, 70);
inline constexpr QColor colorInputBackground = QColor(37, 36, 44);
inline constexpr QColor colorInputBorder = QColor(76, 74, 88);
inline constexpr QColor colorButtonBackground = QColor(58, 57, 80);
inline constexpr QColor colorButtonBorder = QColor(88, 86, 112);
inline constexpr QColor colorButtonHover = QColor(71, 70, 100);
inline constexpr QColor colorButtonPressed = QColor(49, 48, 71);
inline constexpr QColor colorSliderTrack = QColor(90, 88, 103);
inline constexpr QColor colorSliderTrackActive = QColor(217, 215, 227);
inline constexpr QColor colorSliderKnobMask = QColor(45, 44, 52);
inline constexpr QColor colorSliderKnobFill = QColor(43, 42, 50);
inline constexpr QColor colorSliderKnobStroke = QColor(239, 237, 248);
inline constexpr QColor colorNavHintText = QColor(150, 149, 162);

QString px(int value);
void styleLineEdit(QLineEdit* edit);
void styleTitleLabel(QLabel* label);
void styleDescriptionLabel(QLabel* label);
QLineEdit* createNumericEdit(QWidget* parent, double minimum, double maximum, int decimals);
void styleComboBox(QComboBox* combo);
QWidget* buildPanelCardPage(QWidget* parent, QWidget* contentWidget);
void applyNodePanelStyle(QWidget* panel);
QString actionStripStyleSheet();
void styleTitleBar(QFrame* frame);

}
