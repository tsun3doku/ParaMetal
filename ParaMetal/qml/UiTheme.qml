pragma Singleton

import QtQuick

QtObject {
    readonly property color windowBackground: "#202023"
    readonly property color panelBackground: "#2e2e34"
    readonly property color panelRaised: "#2d2c32"
    readonly property color cardBackground: "#2d2c32"
    readonly property color border: "#3a393c"
    readonly property color subtleBorder: "#3c3c45"

    readonly property color text: "#eceaf6"
    readonly property color headingText: "#f6f5fb"
    readonly property color secondaryText: "#d2d1de"
    readonly property color mutedText: "#c8c6d4"

    readonly property color accent: "#5e7cff"
    readonly property color interactiveAccent: "#3578ff"
    readonly property color hover: "#484752"
    readonly property color selected: "#504f5c"

    readonly property color toolNormal: "#3a3a3a"
    readonly property color toolHover: "#505050"
    readonly property color toolPressed: "#5a5a5a"
    readonly property color toolSelectedPressed: "#4b67e0"
    readonly property color toolBorder: "#46464e"

    readonly property color menuBarBackground: "#252529"
    readonly property color menuText: "#d2d2d7"
    readonly property color menuDisabledText: "#6e6e76"
    readonly property color menuSeparator: "#414048"

    readonly property string fontFamily: UiTypography.fontFamily
    readonly property string monoFamily: UiTypography.monoFamily
    readonly property int titleFontSize: UiTypography.titleFontSize
    readonly property int regularFontSize: UiTypography.regularFontSize
    readonly property int regularFontWeight: UiTypography.regularFontWeight
    readonly property int descriptionFontSize: UiTypography.descriptionFontSize
    readonly property int consoleFontSize: UiTypography.consoleFontSize
    readonly property int nodeTitleFontSize: UiTypography.nodeTitleFontSize
    readonly property font regularFont: UiTypography.regularFont
    readonly property font descriptionFont: UiTypography.descriptionFont
    readonly property font nodeTitleFont: UiTypography.nodeTitleFont
}
