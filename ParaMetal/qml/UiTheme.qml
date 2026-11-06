pragma Singleton

import QtQuick

QtObject {
    readonly property color windowBackground: "#1e1e1e"
    readonly property color panelBackground: "#2e2e2e"
    readonly property color splitterCenter: windowBackground
    readonly property color splitterBorder: subtleBorder
    readonly property int splitterWidth: 2
    readonly property int splitterBorderWidth: 2
    readonly property color cardBackground: "#3d3d3d"
    readonly property color border: "#3a3a3a"
    readonly property color subtleBorder: "#3a3a3a"
    readonly property color inputBackground: "#545454"
    readonly property color checkboxBackground: "#545454"
    readonly property color checkboxBorder: "#3a3a3a"

    readonly property color text: "#f0f0f0"
    readonly property color headingText: "#ffffff"
    readonly property color secondaryText: "#b0b0b0"
    readonly property color mutedText: "#808080"
    readonly property color onAccentText: "#ffffff"
    readonly property color activeTabText: "#ffffff"
    readonly property color navigationText: "#a0a0a0"
    readonly property color toolIconColor: "#ffffff"

    readonly property color accent: "#3578ff"
    readonly property color interactiveAccent: "#3578ff"
    readonly property color interactiveHover: "#4a88ff"
    readonly property color hover: "#4a4a4a"
    readonly property color selected: "#545454"

    readonly property color toolNormal: "#343434"
    readonly property color toolHover: "#464646"
    readonly property color toolPressed: "#282828"
    readonly property color toolSelectedPressed: "#2868e0"
    readonly property color toolBorder: "#3a3a3a"
    readonly property color sliderTrack: "#808080"
    readonly property color sliderHandle: "#e0e0e0"
    readonly property color sliderHandleBorder: "#303030"
    readonly property color sourceButton: "#4a4a4a"
    readonly property color sourceButtonHover: "#585858"
    readonly property color sourceButtonBorder: "#3a3a3a"
    readonly property int toolButtonSize: 24
    readonly property int toolButtonPadding: 2
    readonly property int toolButtonFrameInset: 1
    readonly property int toolButtonGroupSpacing: 1
    readonly property int toolButtonFrameSize: toolButtonFrameInset * 2

    function toolButtonGroupWidth(buttonCount) {
        return toolButtonSize * buttonCount
                + toolButtonGroupSpacing * Math.max(0, buttonCount - 1)
                + toolButtonFrameSize
    }

    // Inspector Layout Metrics (Blender-aligned)
    readonly property int inspectorRowSpacing: 6
    readonly property int inspectorBoxSpacing: 1
    readonly property int inspectorLabelSpacing: 6
    readonly property int inspectorGroupSpacing: 10
    readonly property int inspectorLabelWidth: 145
    readonly property int inspectorCompactLabelWidth: 14

    readonly property color menuBarBackground: "#1e1e1e"
    readonly property color menuText: "#f0f0f0"
    readonly property color menuDisabledText: "#707070"
    readonly property color menuSeparator: "#383838"

    readonly property color timelineBackground: "#343434"
    readonly property color timelineTrack: "#808080"
    readonly property color timelineRecorded: "#416d9c"
    readonly property color timelineTick: "#909090"
    readonly property color timelineText: "#c0c0c0"
    readonly property color timelinePlayhead: "#2e7eff"
    readonly property color timelineFrameBackground: interactiveAccent
    readonly property color timelineFrameText: "#ffffff"
    readonly property color timelineFieldBackground: "#404040"
    readonly property color welcomeBackdrop: Qt.rgba(0.04, 0.04, 0.04, 0.80)
    readonly property color welcomeCardBackground: "#2e2e2e"
    readonly property color welcomeTitleBar: "#d0d0d0"
    readonly property color welcomeTitleBarText: "#202020"
    readonly property color welcomeTitleBarBorder: "#a0a0a0"
    readonly property color welcomeOptionBackground: "#1e1e1e"
    readonly property color welcomeOptionHover: "#383838"
    readonly property color welcomeCardBorder: "#3a3a3a"
    readonly property color welcomeText: "#ffffff"
    readonly property color welcomeSubtext: "#b0b0b0"

    readonly property string fontFamily: UiTypography.fontFamily
    readonly property string monoFamily: UiTypography.monoFamily
    readonly property int titleFontSize: UiTypography.titleFontSize
    readonly property int regularFontSize: UiTypography.regularFontSize
    readonly property int regularFontWeight: UiTypography.regularFontWeight
    readonly property int descriptionFontSize: UiTypography.descriptionFontSize
    readonly property int consoleFontSize: UiTypography.consoleFontSize
    readonly property font regularFont: UiTypography.regularFont
    readonly property font descriptionFont: UiTypography.descriptionFont
}
