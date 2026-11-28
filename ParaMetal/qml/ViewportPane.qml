import QtQuick
import ParaMetal

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject bridge
    required property QtObject heatPalette
    required property QtObject velocityPalette
    required property QtObject graphModel
    required property QtObject boundaryFace
    color: theme.windowBackground
    border.width: 1
    border.color: theme.subtleBorder

    ViewportItem {
        objectName: "viewportItem"
        anchors.fill: parent
        anchors.margins: 1
        focus: true
    }

    TemperaturePalette {
        theme: root.theme
        heatPalette: root.heatPalette
        z: 10
    }

    VelocityPalette {
        theme: root.theme
        velocityPalette: root.velocityPalette
        z: 10
    }

    BoundaryFacePanel {
        theme: root.theme
        boundaryFace: root.boundaryFace
        graphModel: root.graphModel
        z: 10
    }

    ViewportTopToolbar {
        id: topToolbar
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.leftMargin: 10
        anchors.rightMargin: 10
        anchors.topMargin: 10
        theme: root.theme
        bridge: root.bridge
        z: 20
    }

    ViewportToolRail {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.leftMargin: 10
        anchors.topMargin: 10
        theme: root.theme
        bridge: root.bridge
        z: 20
    }

    Rectangle {
        id: selectionHint
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 44
        visible: root.bridge.selectionCommitRequired
        radius: 4
        color: root.theme.panelBackground
        border.width: 1
        border.color: root.theme.subtleBorder
        z: 20

        Row {
            id: selectionHintContent
            anchors.centerIn: parent
            spacing: 5

            Text {
                color: root.theme.secondaryText
                font.family: UiTheme.fontFamily
                font.pixelSize: UiTheme.descriptionFontSize
                text: qsTr("Selecting %1")
                    .arg(root.bridge.meshSelectionElementType === 0
                    ? qsTr("Vertices")
                    : root.bridge.meshSelectionElementType === 1
                        ? qsTr("Edges")
                        : qsTr("Faces"))
            }

            Text {
                color: root.theme.accent
                font.family: UiTheme.fontFamily
                font.pixelSize: UiTheme.descriptionFontSize
                font.bold: true
                text: qsTr("Enter")
            }

            Text {
                color: root.theme.secondaryText
                font.family: UiTheme.fontFamily
                font.pixelSize: UiTheme.descriptionFontSize
                text: qsTr("to finish  |")
            }

            Text {
                color: root.theme.accent
                font.family: UiTheme.fontFamily
                font.pixelSize: UiTheme.descriptionFontSize
                font.bold: true
                text: qsTr("Esc")
            }

            Text {
                color: root.theme.secondaryText
                font.family: UiTheme.fontFamily
                font.pixelSize: UiTheme.descriptionFontSize
                text: qsTr("to cancel")
            }
        }

        implicitWidth: selectionHintContent.implicitWidth + 20
        implicitHeight: selectionHintContent.implicitHeight + 12
    }
}
