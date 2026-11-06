import QtQuick
import QtQuick.Controls
import QtQuick.Effects

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
        color: control.enabled
            ? ((control.highlighted || (control.subMenu && control.subMenu.visible))
                ? control.theme.headingText : control.theme.menuText)
            : control.theme.menuDisabledText
        font: control.font
        elide: Text.ElideRight
        horizontalAlignment: Text.AlignLeft
        verticalAlignment: Text.AlignVCenter
    }

    indicator: Item {
        x: control.mirrored ? control.width - width - control.rightPadding : control.leftPadding
        y: control.topPadding + (control.availableHeight - height) / 2
        width: 14
        height: 14
        visible: control.checkable

        Rectangle {
            anchors.fill: parent
            radius: 3
            color: control.checked ? control.theme.accent : control.theme.checkboxBackground
            border.width: 1
            border.color: control.checked ? control.theme.accent : control.theme.checkboxBorder
        }

        Image {
            anchors.fill: parent
            source: "../textures/icons/Settings/check/128w/Artboard 1.png"
            sourceSize.width: width
            sourceSize.height: height
            visible: control.checked
        }
    }

    arrow: Item {
        x: control.mirrored ? control.leftPadding : control.width - width - control.rightPadding
        y: Math.round(control.topPadding + (control.availableHeight - height) / 2)
        width: 8
        height: 8
        visible: control.subMenu !== null

        Image {
            anchors.fill: parent
            source: "../textures/icons/Playback/Next_frame/128w/Artboard 1.png"
            sourceSize.width: 32
            sourceSize.height: 32
            fillMode: Image.PreserveAspectFit
            mirror: control.mirrored
            opacity: control.enabled ? 1.0 : 0.4
            layer.enabled: true
            layer.effect: MultiEffect {
                colorization: 1.0
                colorizationColor: control.enabled
                    ? ((control.highlighted || (control.subMenu && control.subMenu.visible))
                        ? control.theme.headingText : control.theme.menuText)
                    : control.theme.menuDisabledText
            }
        }
    }

    Binding {
        target: control.subMenu
        property: "anchorItem"
        value: (control.subMenu && control.subMenu.visible) ? control : null
        when: control.subMenu !== null
    }

    background: Rectangle {
        implicitWidth: 110
        implicitHeight: 24
        radius: 0
        color: control.down ? control.theme.selected
                            : (control.highlighted || (control.subMenu && control.subMenu.visible))
                              ? control.theme.hover : "transparent"

        // Top border when submenu is open
        Rectangle {
            visible: control.subMenu && control.subMenu.visible
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.rightMargin: -2
            height: 1
            color: control.theme.border
            z: 3
        }

        // Bottom border when submenu is open
        Rectangle {
            visible: control.subMenu && control.subMenu.visible
            anchors.bottom: parent.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.rightMargin: -2
            height: 1
            color: control.theme.border
            z: 3
        }
    }
}
