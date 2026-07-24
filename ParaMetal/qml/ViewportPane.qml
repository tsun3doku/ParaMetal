import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import ParaMetal

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject bridge
    required property QtObject heatPalette
    color: theme.windowBackground

    ViewportItem {
        id: viewport
        objectName: "viewportItem"
        anchors.fill: parent
        focus: true
    }

    TemperaturePalette {
        id: temperaturePalette
        theme: root.theme
        heatPalette: root.heatPalette
        z: 10
    }

    Row {
        anchors.top: parent.top
        anchors.topMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter
        spacing: 5

        Rectangle {
            width: 67
            height: 34
            radius: 5
            color: theme.toolBorder
            Row {
                x: 1
                y: 1
                spacing: 1
                ToolIconButton {
                    width: 32; height: 32; segment: 1
                    iconSize: 20
                    baseColor: theme.toolNormal; hoverColor: theme.toolHover; pressedColor: theme.toolPressed
                    imageSource: "../textures/icons/Overlays/wireframe/32w/Artboard 1.png"
                    checked: bridge.wireframeMode === 1
                    onClicked: bridge.setWireframeMode(checked ? 1 : 0)
                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("Wireframe")
                }
                ToolIconButton {
                    width: 32; height: 32; segment: 3
                    iconSize: 20
                    baseColor: theme.toolNormal; hoverColor: theme.toolHover; pressedColor: theme.toolPressed
                    imageSource: "../textures/icons/Overlays/wireframe_shaded/32w/Artboard 1.png"
                    checked: bridge.wireframeMode === 2
                    onClicked: bridge.setWireframeMode(checked ? 2 : 0)
                    ToolTip.visible: hovered
                    ToolTip.text: qsTr("Shaded wireframe")
                }
            }
        }

        Rectangle {
            width: 34
            height: 34
            radius: 5
            color: theme.toolBorder
            ToolIconButton {
                x: 1; y: 1; width: 32; height: 32
                iconSize: 20
                baseColor: theme.toolNormal; hoverColor: theme.toolHover; pressedColor: theme.toolPressed
                imageSource: "../textures/icons/Overlays/grid/32w/Artboard 1.png"
                checked: bridge.gridEnabled
                onClicked: bridge.setGridEnabled(checked)
                ToolTip.visible: hovered
                ToolTip.text: qsTr("Grid")
            }
        }
    }
}
