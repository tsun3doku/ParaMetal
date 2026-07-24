import QtQuick
import QtQuick.Controls

MenuItem {
    id: control

    required property QtObject theme

    implicitWidth: Math.max(implicitBackgroundWidth + leftInset + rightInset, implicitContentWidth + leftPadding + rightPadding)
    implicitHeight: Math.max(implicitBackgroundHeight + topInset + bottomInset, implicitContentHeight + topPadding + bottomPadding)
    leftPadding: 12
    rightPadding: 10
    topPadding: 5
    bottomPadding: 5
    spacing: 6
    font.family: theme.fontFamily
    font.pixelSize: theme.regularFontSize
    font.weight: Font.Light

    contentItem: Text {
        readonly property real indicatorPadding: control.checkable && control.indicator
                                                 ? control.indicator.width + control.spacing : 0
        readonly property real arrowPadding: control.subMenu && control.arrow
                                             ? control.arrow.width + control.spacing : 0

        leftPadding: control.mirrored ? arrowPadding : indicatorPadding
        rightPadding: control.mirrored ? indicatorPadding : arrowPadding
        text: control.text
        color: control.enabled ? control.theme.menuText : control.theme.menuDisabledText
        font: control.font
        elide: Text.ElideRight
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
    }

    indicator: Text {
        x: control.mirrored ? control.width - width - control.rightPadding : control.leftPadding
        y: control.topPadding + (control.availableHeight - height) / 2
        width: 10
        height: 14
        visible: control.checkable
        text: control.checked ? "\u2713" : ""
        color: control.theme.menuText
        font.family: control.theme.fontFamily
        font.pixelSize: control.theme.regularFontSize
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    arrow: Text {
        x: control.mirrored ? control.leftPadding : control.width - width - control.rightPadding
        y: control.topPadding + (control.availableHeight - height) / 2
        width: 10
        height: 14
        visible: control.subMenu
        text: control.mirrored ? "\u2039" : "\u203a"
        color: control.enabled ? control.theme.menuText : control.theme.menuDisabledText
        font.family: control.theme.fontFamily
        font.pixelSize: 17
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    background: Rectangle {
        implicitWidth: 110
        implicitHeight: 26
        color: control.down ? control.theme.selected
                            : control.highlighted ? control.theme.hover : "transparent"
    }
}
