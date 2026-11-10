pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

CheckBox {
    id: control

    required property QtObject theme

    implicitHeight: 26
    Layout.fillWidth: true
    leftPadding: 24
    rightPadding: 4
    font: theme.regularFont
    hoverEnabled: true

    indicator: Item {
        x: 4
        y: (control.height - height) / 2
        width: 14
        height: 14
        opacity: control.enabled ? 1.0 : 0.4

        Rectangle {
            anchors.fill: parent
            radius: 3
            color: control.checked
                   ? (control.hovered ? control.theme.interactiveHover : control.theme.accent)
                   : (control.hovered ? Qt.lighter(control.theme.checkboxBackground, 1.30) : control.theme.checkboxBackground)
            border.width: 1
            border.color: control.checked
                          ? (control.hovered ? control.theme.interactiveHover : control.theme.accent)
                          : (control.hovered ? Qt.lighter(control.theme.checkboxBorder, 1.45) : control.theme.checkboxBorder)

            Behavior on color {
                ColorAnimation { duration: 40; easing.type: Easing.OutQuad }
            }
            Behavior on border.color {
                ColorAnimation { duration: 40; easing.type: Easing.OutQuad }
            }
        }

        Image {
            anchors.fill: parent
            source: "../textures/icons/Settings/check/128w/Artboard 1.png"
            sourceSize.width: width
            sourceSize.height: height
            smooth: true
            visible: control.checked
        }
    }

    contentItem: Text {
        text: control.text
        color: control.theme.text
        font: control.font
        verticalAlignment: Text.AlignVCenter
    }

    background: Rectangle {
        radius: 3
        color: "transparent"
    }
}
