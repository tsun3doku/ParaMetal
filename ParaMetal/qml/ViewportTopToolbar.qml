import QtQuick

Item {
    id: root

    required property QtObject theme
    required property QtObject bridge

    implicitHeight: UiTheme.toolButtonSize + UiTheme.toolButtonFrameSize

    Row {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        spacing: 5

        Rectangle {
            width: UiTheme.toolButtonGroupWidth(2)
            height: root.height
            radius: 5
            color: root.theme.toolBorder

            Row {
                x: UiTheme.toolButtonFrameInset
                y: UiTheme.toolButtonFrameInset
                spacing: UiTheme.toolButtonGroupSpacing

                ToolIconButton {
                    segment: 1
                    iconSize: 20
                    imageSource: "../textures/icons/Overlays/wireframe/128w/Artboard 1.png"
                    checked: root.bridge.wireframeMode === 1
                    onClicked: root.bridge.setWireframeMode(checked ? 1 : 0)
                }

                ToolIconButton {
                    segment: 3
                    iconSize: 20
                    imageSource: "../textures/icons/Overlays/wireframe_shaded/128w/Artboard 1.png"
                    checked: root.bridge.wireframeMode === 2
                    onClicked: root.bridge.setWireframeMode(checked ? 2 : 0)
                }
            }
        }

        Rectangle {
            width: UiTheme.toolButtonSize + UiTheme.toolButtonFrameSize
            height: root.height
            radius: 5
            color: root.theme.toolBorder

            ToolIconButton {
                x: UiTheme.toolButtonFrameInset
                y: UiTheme.toolButtonFrameInset
                iconSize: 20
                imageSource: "../textures/icons/Overlays/grid/128w/Artboard 1.png"
                checked: root.bridge.gridEnabled
                onClicked: root.bridge.setGridEnabled(checked)
            }
        }

    }

    Row {
        anchors.right: parent.right
        anchors.rightMargin: 8
        anchors.top: parent.top
        spacing: 5

        CameraControl {
            theme: root.theme
            bridge: root.bridge
        }

        ViewportSettingsControl {
            theme: root.theme
            bridge: root.bridge
        }
    }
}
