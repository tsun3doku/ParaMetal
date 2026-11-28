import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root

    required property QtObject theme
    required property QtObject bridge

    width: UiTheme.toolButtonGroupWidth(2)
    height: UiTheme.toolButtonSize + UiTheme.toolButtonFrameSize

    Rectangle {
        anchors.fill: parent
        radius: 5
        color: root.theme.toolBorder

        Row {
            x: UiTheme.toolButtonFrameInset
            y: UiTheme.toolButtonFrameInset
            spacing: UiTheme.toolButtonGroupSpacing

            ToolIconButton {
                segment: 1
                checkable: false
                iconSize: 20
                imageSource: "../textures/icons/Overlays/camera/128w/Artboard 1.png"
                onClicked: root.bridge.focusWorldOrigin()
            }

            ToolIconButton {
                segment: 3
                checkable: false
                iconSize: 14
                imageSource: "../textures/icons/Overlays/arrow/128w/Artboard 1.png"
                onClicked: cameraPopup.visible ? cameraPopup.close() : cameraPopup.open()
            }
        }
    }

    UiPanelPopup {
        id: cameraPopup
        theme: root.theme
        x: root.width - width
        y: root.height + 4
        width: 205

        ButtonGroup { id: projectionGroup }

        RowLayout {
            Layout.fillWidth: true
            spacing: 1

            Button {
                id: perspectiveButton
                Layout.fillWidth: true
                implicitHeight: 22
                checkable: true
                ButtonGroup.group: projectionGroup
                checked: root.bridge.projectionMode === 0
                text: qsTr("Perspective")
                onClicked: root.bridge.setProjectionMode(0)
                contentItem: Text {
                    text: perspectiveButton.text
                    color: perspectiveButton.checked ? root.theme.onAccentText : root.theme.text
                    font: root.theme.regularFont
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    topLeftRadius: 3
                    bottomLeftRadius: 3
                    color: perspectiveButton.checked ? root.theme.interactiveAccent
                                                     : perspectiveButton.hovered ? root.theme.toolHover
                                                                                 : root.theme.toolNormal
                }
            }

            Button {
                id: orthographicButton
                Layout.fillWidth: true
                implicitHeight: 22
                checkable: true
                ButtonGroup.group: projectionGroup
                checked: root.bridge.projectionMode === 1
                text: qsTr("Orthographic")
                onClicked: root.bridge.setProjectionMode(1)
                contentItem: Text {
                    text: orthographicButton.text
                    color: orthographicButton.checked ? root.theme.onAccentText : root.theme.text
                    font: root.theme.regularFont
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    topRightRadius: 3
                    bottomRightRadius: 3
                    color: orthographicButton.checked ? root.theme.interactiveAccent
                                                      : orthographicButton.hovered ? root.theme.toolHover
                                                                                   : root.theme.toolNormal
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            enabled: root.bridge.projectionMode === 0
            opacity: enabled ? 1.0 : 0.45

            Text {
                Layout.preferredWidth: 80
                text: qsTr("FOV")
                color: root.theme.text
                font: root.theme.regularFont
                elide: Text.ElideRight
            }

            UiNumberBox {
                Layout.fillWidth: true
                Layout.preferredHeight: 22
                theme: root.theme
                value: root.bridge.cameraFov
                minimum: 1.0
                maximum: 120.0
                decimals: 1
                onValueEdited: val => root.bridge.setCameraFov(val)
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                Layout.preferredWidth: 80
                text: qsTr("Zoom Speed")
                color: root.theme.text
                font: root.theme.regularFont
                elide: Text.ElideRight
            }

            UiNumberBox {
                Layout.fillWidth: true
                Layout.preferredHeight: 22
                theme: root.theme
                value: root.bridge.cameraZoomSpeed
                minimum: 0.1
                maximum: 5.0
                decimals: 1
                onValueEdited: val => root.bridge.setCameraZoomSpeed(val)
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                Layout.preferredWidth: 80
                text: qsTr("Pan Speed")
                color: root.theme.text
                font: root.theme.regularFont
                elide: Text.ElideRight
            }

            UiNumberBox {
                Layout.fillWidth: true
                Layout.preferredHeight: 22
                theme: root.theme
                value: root.bridge.cameraPanSpeed
                minimum: 0.1
                maximum: 5.0
                decimals: 1
                onValueEdited: val => root.bridge.setCameraPanSpeed(val)
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Text {
                Layout.preferredWidth: 80
                text: qsTr("Cull")
                color: root.theme.text
                font: root.theme.regularFont
                elide: Text.ElideRight
            }

            UiNumberBox {
                Layout.fillWidth: true
                Layout.preferredHeight: 22
                theme: root.theme
                value: root.bridge.nearClip
                minimum: 0.01
                maximum: 10.0
                decimals: 2
                onValueEdited: val => root.bridge.setNearClip(val)
            }
        }

        UiCheckBox {
            theme: root.theme
            text: qsTr("Cull streamlines only")
            checked: root.bridge.cullLinesOnly
            onClicked: root.bridge.setCullLinesOnly(checked)
        }
    }
}
