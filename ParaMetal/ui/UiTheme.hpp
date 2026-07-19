#pragma once

#include <QColor>
#include <QSplitter>

#include "UiTypography.hpp"

#include <string>

namespace ui {

inline constexpr QColor PanelBackground = QColor(46, 46, 52);

inline constexpr int SplitterVisualThickness = 7;
inline constexpr QColor SplitterBaseColor = QColor(60, 60, 69);
inline constexpr QColor SplitterHoverColor = QColor(60, 60, 69);
inline constexpr QColor SplitterBorderColor = QColor(60, 60, 69);

inline constexpr QColor MenuBarBackground = QColor(36, 36, 41);
inline constexpr QColor MenuBarText = QColor(210, 210, 215);
inline constexpr QColor MenuBarItemHover = QColor(58, 57, 65);
inline constexpr QColor MenuBackground = PanelBackground;
inline constexpr QColor MenuBorder = QColor(70, 70, 78);
inline constexpr QColor MenuItemHover = QColor(72, 71, 82);
inline constexpr QColor MenuItemSelected = QColor(80, 79, 92);
inline constexpr QColor MenuText = QColor(210, 210, 215);
inline constexpr QColor MenuTextDisabled = QColor(110, 110, 118);
inline constexpr QColor MenuSeparator = QColor(65, 64, 72);
inline constexpr int ScrollBarThickness = 12;
inline constexpr int ScrollBarMargin = 2;
inline constexpr int ScrollBarRadius = 4;
inline constexpr int ScrollBarMinHandleLength = 28;
inline constexpr QColor ScrollBarTrackColor = QColor(70, 70, 80);
inline constexpr QColor ScrollBarHandleColor = QColor(100, 100, 110);
inline constexpr QColor ScrollBarHandleHoverColor = QColor(180, 180, 180);

inline constexpr QColor InteractiveAccent = QColor(53, 120, 255);

inline constexpr QColor ToolButtonNormal = QColor(58, 58, 58);       
inline constexpr QColor ToolButtonHover = QColor(80, 80, 80);        
inline constexpr QColor ToolButtonPressed = QColor(90, 90, 90);      
inline constexpr QColor ToolButtonSelected = InteractiveAccent;
inline constexpr QColor ToolButtonSelectedPressed = QColor(75, 103, 224);
inline constexpr QColor ToolButtonBorder = QColor(70, 70, 78);
inline constexpr QColor ToolButtonDisabled = QColor(37, 37, 37);     
inline constexpr QColor ToolButtonDisabledText = QColor(85, 85, 85); 
inline constexpr QColor IconDefault = QColor(221, 221, 221);         

inline constexpr QColor TimelineBackground = QColor(30, 30, 30);     
inline constexpr QColor TimelineTrackBg = QColor(51, 51, 51);       
inline constexpr QColor TimelineTrackFill = QColor(65, 109, 156);    
inline constexpr QColor TimelineText = QColor(204, 204, 204);        
inline constexpr QColor TimelineTextDim = QColor(102, 102, 102);    

enum class ToolButtonSegment {
    Standalone,
    Leading,
    Middle,
    Trailing
};

inline std::string toolButtonStyle(ToolButtonSegment segment = ToolButtonSegment::Standalone) {
    std::string radius;
    switch (segment) {
        case ToolButtonSegment::Leading:
            radius = "border-top-left-radius: 4px; border-bottom-left-radius: 4px;";
            break;
        case ToolButtonSegment::Trailing:
            radius = "border-top-right-radius: 4px; border-bottom-right-radius: 4px;";
            break;
        case ToolButtonSegment::Middle:
            radius = "border-radius: 0px;";
            break;
        default:
            radius = "border-radius: 4px;";
            break;
    }
    return
        "QPushButton {"
        "  background-color: " + ToolButtonNormal.name().toStdString() + ";"
        "  border: none;"
        "  " + radius +
        "  color: " + IconDefault.name().toStdString() + ";"
        "}"
        "QPushButton:hover:!checked {"
        "  background-color: " + ToolButtonHover.name().toStdString() + ";"
        "}"
        "QPushButton:pressed {"
        "  background-color: " + ToolButtonPressed.name().toStdString() + ";"
        "}"
        "QPushButton:checked {"
        "  background-color: " + ToolButtonSelected.name().toStdString() + ";"
        "}"
        "QPushButton:checked:pressed {"
        "  background-color: " + ToolButtonSelectedPressed.name().toStdString() + ";"
        "}"
        "QPushButton:disabled {"
        "  background-color: " + ToolButtonDisabled.name().toStdString() + ";"
        "  color: " + ToolButtonDisabledText.name().toStdString() + ";"
        "}";
}

inline std::string appStyleSheet() {
    const std::string splitterThickness = std::to_string(SplitterVisualThickness);
    const std::string scrollBarThickness = std::to_string(ScrollBarThickness);
    const std::string scrollBarMargin = std::to_string(ScrollBarMargin);
    const std::string scrollBarRadius = std::to_string(ScrollBarRadius);
    const std::string scrollBarMinHandleLength = std::to_string(ScrollBarMinHandleLength);
    return
        "QSplitter::handle:horizontal { "
        "width: " + splitterThickness + "px; "
        "background: " + SplitterBaseColor.name().toStdString() + "; "
        "border-left: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "border-right: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "}"
        "QSplitter::handle:horizontal:hover { "
        "background: " + SplitterHoverColor.name().toStdString() + "; "
        "}"
        "QSplitter::handle:vertical { "
        "height: " + splitterThickness + "px; "
        "background: " + SplitterBaseColor.name().toStdString() + "; "
        "border-top: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "border-bottom: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "}"
        "QSplitter::handle:vertical:hover { "
        "background: " + SplitterHoverColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator { "
        "background: " + SplitterBaseColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator:hover { "
        "background: " + SplitterHoverColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator:horizontal { "
        "background: " + SplitterBaseColor.name().toStdString() + "; "
        "width: " + splitterThickness + "px; "
        "border-left: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "border-right: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator:horizontal:hover { "
        "background: " + SplitterHoverColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator:vertical { "
        "background: " + SplitterBaseColor.name().toStdString() + "; "
        "height: " + splitterThickness + "px; "
        "border-top: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "border-bottom: 1px solid " + SplitterBorderColor.name().toStdString() + "; "
        "}"
        "QMainWindow::separator:vertical:hover { "
        "background: " + SplitterHoverColor.name().toStdString() + "; "
        "}"
        "QDockWidget { "
        "border: 0px; "
        "}"
        "QDockWidget > QWidget { "
        "border: 0px; "
        "}"
        "QScrollBar:vertical { "
        "background: " + ScrollBarTrackColor.name().toStdString() + "; "
        "width: " + scrollBarThickness + "px; "
        "margin: " + scrollBarMargin + "px 0px " + scrollBarMargin + "px 0px; "
        "border: none; "
        "}"
        "QScrollBar::handle:vertical { "
        "background: " + ScrollBarHandleColor.name().toStdString() + "; "
        "min-height: " + scrollBarMinHandleLength + "px; "
        "border-radius: " + scrollBarRadius + "px; "
        "}"
        "QScrollBar::handle:vertical:hover { "
        "background: " + ScrollBarHandleHoverColor.name().toStdString() + "; "
        "}"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { "
        "height: 0px; "
        "background: transparent; "
        "border: none; "
        "}"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { "
        "background: transparent; "
        "}"
        "QScrollBar:horizontal { "
        "background: " + ScrollBarTrackColor.name().toStdString() + "; "
        "height: " + scrollBarThickness + "px; "
        "margin: 0px " + scrollBarMargin + "px 0px " + scrollBarMargin + "px; "
        "border: none; "
        "}"
        "QScrollBar::handle:horizontal { "
        "background: " + ScrollBarHandleColor.name().toStdString() + "; "
        "min-width: " + scrollBarMinHandleLength + "px; "
        "border-radius: " + scrollBarRadius + "px; "
        "}"
        "QScrollBar::handle:horizontal:hover { "
        "background: " + ScrollBarHandleHoverColor.name().toStdString() + "; "
        "}"
        "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { "
        "width: 0px; "
        "background: transparent; "
        "border: none; "
        "}"
        "QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { "
        "background: transparent; "
        "}"
        "QMenuBar { "
        "background: " + MenuBarBackground.name().toStdString() + "; "
        "color: " + MenuBarText.name().toStdString() + "; "
        "border-bottom: 1px solid " + MenuBorder.name().toStdString() + "; "
        "}"
        "QMenuBar::item { "
        "background: transparent; "
        "padding: 4px 10px; "
        "}"
        "QMenuBar::item:selected { "
        "background: " + MenuBarItemHover.name().toStdString() + "; "
        "border-radius: 3px; "
        "}"
        "QMenuBar::item:pressed { "
        "background: " + MenuItemSelected.name().toStdString() + "; "
        "border-radius: 3px; "
        "}"
        "QMenu { "
        "background: " + MenuBackground.name().toStdString() + "; "
        "color: " + MenuText.name().toStdString() + "; "
        "border: 1px solid " + MenuBorder.name().toStdString() + "; "
        "padding: 3px 0px; "
        "}"
        "QMenu::item { "
        "padding: 5px 28px 5px 20px; "
        "background: transparent; "
        "}"
        "QMenu::item:selected { "
        "background: " + MenuItemHover.name().toStdString() + "; "
        "}"
        "QMenu::item:disabled { "
        "color: " + MenuTextDisabled.name().toStdString() + "; "
        "}"
        "QMenu::separator { "
        "height: 1px; "
        "background: " + MenuSeparator.name().toStdString() + "; "
        "margin: 3px 8px; "
        "}"
        "QMenu::indicator { "
        "width: 14px; "
        "height: 14px; "
        "margin-left: 4px; "
        "}";
}

inline void configureSplitter(QSplitter& splitter) {
    splitter.setHandleWidth(SplitterVisualThickness);
    splitter.setOpaqueResize(true);
}

}
