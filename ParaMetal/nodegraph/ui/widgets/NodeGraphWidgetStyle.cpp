#include "NodeGraphWidgetStyle.hpp"
#include "ui/UiTheme.hpp"
#include "ui/UiTypography.hpp"

#include <QComboBox>
#include <QFont>
#include <QFrame>
#include <QLabel>
#include <QLineEdit>
#include <QString>
#include <QVBoxLayout>
#include <QWidget>

namespace nodegraphwidgets {

QString px(int value) {
    return QString::number(value) + "px";
}

void styleLineEdit(QLineEdit* edit) {
    if (!edit) {
        return;
    }

    edit->setFrame(false);
    edit->setFont(ui::UiTypography::font(ui::TextRole::Regular));

    QString stylesheet;
    stylesheet += "QLineEdit {";
    stylesheet += "  background: " + colorInputBackground.name() + ";";
    stylesheet += "  color: " + colorTextInput.name() + ";";
    stylesheet += "  border: 1px solid " + colorInputBorder.name() + ";";
    stylesheet += "  border-radius: " + px(panelWidgetRadius) + ";";
    stylesheet += "  padding: "
        + px(panelFieldVerticalPadding) + " "
        + px(panelFieldHorizontalPadding) + ";";
    stylesheet += "  selection-background-color: " + colorAccent.name() + ";";
    stylesheet += "}";
    stylesheet += "QLineEdit:focus {";
    stylesheet += "  border: 1px solid " + colorAccentFocus.name() + ";";
    stylesheet += "}";
    edit->setStyleSheet(stylesheet);
}

QLineEdit* createNumericEdit(QWidget* parent, double minimum, double maximum, int decimals) {
    QLineEdit* edit = new QLineEdit(parent);
    styleLineEdit(edit);
    auto* validator = new QDoubleValidator(minimum, maximum, decimals, edit);
    validator->setNotation(QDoubleValidator::StandardNotation);
    edit->setValidator(validator);
    return edit;
}

void styleComboBox(QComboBox* combo) {
    if (!combo) {
        return;
    }

    combo->setFont(ui::UiTypography::font(ui::TextRole::Regular));

    QString stylesheet;
    stylesheet += "QComboBox {";
    stylesheet += "  background: " + colorInputBackground.name() + ";";
    stylesheet += "  color: " + colorTextInput.name() + ";";
    stylesheet += "  border: 1px solid " + colorInputBorder.name() + ";";
    stylesheet += "  border-radius: " + px(panelWidgetRadius) + ";";
    stylesheet += "  padding: "
        + px(panelFieldVerticalPadding) + " "
        + px(panelFieldHorizontalPadding) + ";";
    stylesheet += "}";
    stylesheet += "QComboBox:focus {";
    stylesheet += "  border: 1px solid " + colorAccentFocus.name() + ";";
    stylesheet += "}";
    stylesheet += "QComboBox::drop-down {";
    stylesheet += "  width: 24px;";
    stylesheet += "  border: none;";
    stylesheet += "}";
    stylesheet += "QComboBox::down-arrow {";
    stylesheet += "  width: 8px;";
    stylesheet += "  height: 8px;";
    stylesheet += "}";
    stylesheet += "QComboBox QAbstractItemView {";
    stylesheet += "  background: " + colorInputBackground.name() + ";";
    stylesheet += "  color: " + colorTextInput.name() + ";";
    stylesheet += "  border: 1px solid " + colorInputBorder.name() + ";";
    stylesheet += "  selection-background-color: " + colorAccent.name() + ";";
    stylesheet += "}";
    combo->setStyleSheet(stylesheet);
}

QWidget* buildPanelCardPage(QWidget* parent, QWidget* contentWidget) {
    QWidget* page = new QWidget(parent);
    QVBoxLayout* pageLayout = new QVBoxLayout(page);
    pageLayout->setContentsMargins(0, 0, 0, 0);

    QFrame* card = new QFrame(page);
    card->setObjectName("NodePanelCard");
    QVBoxLayout* cardLayout = new QVBoxLayout(card);
    cardLayout->setContentsMargins(
        panelMarginLeft,
        panelMarginTop,
        panelMarginRight,
        panelMarginBottom);
    cardLayout->setSpacing(panelCardInnerSpacing);
    cardLayout->addWidget(contentWidget);

    pageLayout->addWidget(card);
    return page;
}

void applyNodePanelStyle(QWidget* panel) {
    panel->setObjectName("NodePanelRoot");
    panel->setFont(ui::UiTypography::font(ui::TextRole::Regular));

    QString stylesheet;
    stylesheet += "QWidget#NodePanelRoot {";
    stylesheet += "  background: " + colorPanelBackground.name() + ";";
    stylesheet += "  color: " + colorTextPrimary.name() + ";";
    stylesheet += "}";
    stylesheet += "QScrollArea {";
    stylesheet += "  border: none;";
    stylesheet += "  background: transparent;";
    stylesheet += "}";
    stylesheet += "QWidget#NodePanelContent {";
    stylesheet += "  background: " + colorPanelBackground.name() + ";";
    stylesheet += "}";
    stylesheet += "QFrame#NodePanelCard {";
    stylesheet += "  background: " + colorPanelCardBackground.name() + ";";
    stylesheet += "  border: 1px solid " + colorPanelCardBorder.name() + ";";
    stylesheet += "  border-radius: " + px(panelCardRadius) + ";";
    stylesheet += "}";
    stylesheet += "QLabel#NodePanelTitle {";
    stylesheet += "  color: " + colorTextHeading.name() + ";";
    stylesheet += "}";
    stylesheet += "QLabel#NodePanelSubtitle {";
    stylesheet += "  color: " + colorTextSecondary.name() + ";";
    stylesheet += "}";
    stylesheet += "QLabel#NodePanelStatus {";
    stylesheet += "  color: " + colorStatusAccent.name() + ";";
    stylesheet += "  padding-top: " + px(panelStatusTopPadding) + ";";
    stylesheet += "}";
    stylesheet += "QLabel#NodePanelDescription {";
    stylesheet += "  color: " + colorTextSecondary.name() + ";";
    stylesheet += "}";
    stylesheet += "QTabWidget::pane {";
    stylesheet += "  border: none;";
    stylesheet += "  background: transparent;";
    stylesheet += "  top: -1px;";
    stylesheet += "}";
    stylesheet += "QTabBar::tab {";
    stylesheet += "  background: transparent;";
    stylesheet += "  color: " + colorTextMuted.name() + ";";
    stylesheet += "  border: none;";
    stylesheet += "  padding: "
        + px(panelTabTopPadding) + " "
        + px(panelTabHorizontalPadding) + " "
        + px(panelTabBottomPadding) + " "
        + px(panelTabHorizontalPadding) + ";";
    stylesheet += "  margin-right: " + px(panelTabRightMargin) + ";";
    stylesheet += "}";
    stylesheet += "QTabBar::tab:selected {";
    stylesheet += "  color: " + colorTextTabSelected.name() + ";";
    stylesheet += "  border-bottom: "
        + px(panelTabSelectedBorderWidth) + " solid "
        + colorAccent.name() + ";";
    stylesheet += "}";
    stylesheet += "QTabBar::tab:hover:!selected {";
    stylesheet += "  color: " + colorTextHover.name() + ";";
    stylesheet += "}";
    stylesheet += "QLabel {";
    stylesheet += "  color: " + colorTextPrimary.name() + ";";
    stylesheet += "}";
    stylesheet += "QCheckBox {";
    stylesheet += "  color: " + colorTextPrimary.name() + ";";
    stylesheet += "  spacing: " + px(panelCheckboxSpacing) + ";";
    stylesheet += "}";
    stylesheet += "QCheckBox::indicator {";
    stylesheet += "  width: " + px(panelCheckboxIndicatorSize) + ";";
    stylesheet += "  height: " + px(panelCheckboxIndicatorSize) + ";";
    stylesheet += "  border-radius: " + px(panelCheckboxIndicatorRadius) + ";";
    stylesheet += "  border: 1px solid " + colorCheckboxBorder.name() + ";";
    stylesheet += "  background: " + colorCheckboxBackground.name() + ";";
    stylesheet += "}";
    stylesheet += "QCheckBox::indicator:checked {";
    stylesheet += "  background: " + colorAccent.name() + ";";
    stylesheet += "  border: 1px solid " + colorAccent.name() + ";";
    stylesheet += "}";
    stylesheet += "QLineEdit, QComboBox, QTextEdit, QPlainTextEdit {";
    stylesheet += "  background: " + colorInputBackground.name() + ";";
    stylesheet += "  color: " + colorTextInput.name() + ";";
    stylesheet += "  border: 1px solid " + colorInputBorder.name() + ";";
    stylesheet += "  border-radius: " + px(panelWidgetRadius) + ";";
    stylesheet += "  padding: "
        + px(panelFieldVerticalPadding) + " "
        + px(panelFieldHorizontalPadding) + ";";
    stylesheet += "  selection-background-color: " + colorAccent.name() + ";";
    stylesheet += "}";
    stylesheet += "QLineEdit:focus, QComboBox:focus, QTextEdit:focus {";
    stylesheet += "  border: 1px solid " + colorAccentFocus.name() + ";";
    stylesheet += "}";
    stylesheet += "QComboBox {";
    stylesheet += "  padding-right: 24px;";
    stylesheet += "}";
    stylesheet += "QComboBox::drop-down {";
    stylesheet += "  width: 24px;";
    stylesheet += "  border: none;";
    stylesheet += "}";
    stylesheet += "QComboBox::down-arrow {";
    stylesheet += "  width: 8px;";
    stylesheet += "  height: 8px;";
    stylesheet += "}";
    stylesheet += "QPushButton {";
    stylesheet += "  background: " + colorButtonBackground.name() + ";";
    stylesheet += "  color: " + colorTextHeading.name() + ";";
    stylesheet += "  border: 1px solid " + colorButtonBorder.name() + ";";
    stylesheet += "  border-radius: " + px(panelWidgetRadius) + ";";
    stylesheet += "  padding: "
        + px(panelButtonVerticalPadding) + " "
        + px(panelButtonHorizontalPadding) + ";";
    stylesheet += "}";
    stylesheet += "QPushButton:hover {";
    stylesheet += "  background: " + colorButtonHover.name() + ";";
    stylesheet += "}";
    stylesheet += "QPushButton:pressed {";
    stylesheet += "  background: " + colorButtonPressed.name() + ";";
    stylesheet += "}";
    panel->setStyleSheet(stylesheet);
}

QString actionStripStyleSheet() {
    QString ss;
    ss += "QWidget#NodeGraphActionStrip {";
    ss += "  background: " + colorPanelCardBackground.name() + ";";
    ss += "  border: 1px solid " + colorInputBorder.name() + ";";
    ss += "  border-radius: " + px(actionStripBannerRadius) + ";";
    ss += "}";
    ss += "QLabel#ActionStripTitle {";
    ss += "  color: " + colorTextHeading.name() + ";";
    ss += "}";
    ss += "QLabel#ActionStripDescription {";
    ss += "  color: " + colorTextSecondary.name() + ";";
    ss += "}";
    ss += "QPushButton#ActionStripButton {";
    ss += "  background: " + ui::InteractiveAccent.name() + ";";
    ss += "  color: " + colorTextOnAccent.name() + ";";
    ss += "  border: 1px solid " + ui::InteractiveAccent.name() + ";";
    ss += "  border-radius: " + px(panelWidgetRadius) + ";";
    ss += "  padding: 6px 10px;";
    ss += "}";
    ss += "QPushButton#ActionStripButton:hover {";
    ss += "  background: " + colorActionStripButtonHover.name() + ";";
    ss += "  border-color: " + colorActionStripButtonHover.name() + ";";
    ss += "}";
    ss += "QPushButton#ActionStripDismiss {";
    ss += "  background: transparent;";
    ss += "  border: none;";
    ss += "  border-radius: " + px(panelWidgetRadius) + ";";
    ss += "  padding: 0px;";
    ss += "}";
    ss += "QPushButton#ActionStripDismiss:hover {";
    ss += "  background: " + colorButtonHover.name() + ";";
    ss += "}";
    return ss;
}

void styleTitleBar(QFrame* frame) {
    if (!frame) {
        return;
    }
    frame->setFont(ui::UiTypography::font(ui::TextRole::Title));
    frame->setFixedHeight(panelTitleBarHeight);
    QString ss;
    ss += "QFrame {";
    ss += "  background: " + colorPanelBackground.name() + ";";
    ss += "  border-bottom: 1px solid " + colorPanelCardBorder.name() + ";";
    ss += "}";
    ss += "QLabel {";
    ss += "  color: " + colorTextHeading.name() + ";";
    ss += "}";
    frame->setStyleSheet(ss);
}

}

void nodegraphwidgets::styleTitleLabel(QLabel* label) {
    if (!label) {
        return;
    }
    label->setFont(ui::UiTypography::font(ui::TextRole::Title));
}

void nodegraphwidgets::styleDescriptionLabel(QLabel* label) {
    if (!label) {
        return;
    }
    label->setObjectName(QStringLiteral("NodePanelDescription"));
    label->setFont(ui::UiTypography::font(ui::TextRole::Description));
}
