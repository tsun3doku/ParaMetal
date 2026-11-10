pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject panelModel

    property string title
    property int panelWidth: 156
    property int expandedHeight: 286
    readonly property int headerHeight: 24
    property bool minimized: false

    default property alias contentData: body.data

    visible: panelModel.visible
    width: panelWidth
    height: minimized ? headerHeight : expandedHeight
    radius: 4
    color: theme.panelBackground
    border.color: theme.border

    Binding {
        target: root
        property: "x"
        value: panelModel.x >= 0 ? panelModel.x : 16
        when: !headerDrag.drag.active
    }
    Binding {
        target: root
        property: "y"
        value: root.minimized
            ? root.parent.height - root.headerHeight
            : (panelModel.y >= 0 ? panelModel.y : root.parent.height - root.expandedHeight - 16)
        when: !headerDrag.drag.active
    }

    Rectangle {
        id: header
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        height: root.headerHeight
        color: root.theme.panelHeaderBackground
        radius: root.radius
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: parent.radius
            color: parent.color
        }

        Text {
            anchors.verticalCenter: parent.verticalCenter
            x: 10
            text: root.title
            color: root.theme.headingText
            font: root.theme.regularFont
        }

        Image {
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            anchors.rightMargin: 8
            width: 14
            height: 14
            source: "../textures/icons/Overlays/arrow/128w/Artboard 1.png"
            sourceSize.width: 14
            sourceSize.height: 14
            smooth: true
            fillMode: Image.PreserveAspectFit
            rotation: root.minimized ? 180 : 0
            opacity: collapseArea.containsMouse ? 1.0 : 0.85
        }

        MouseArea {
            id: headerDrag
            anchors.fill: parent
            enabled: !root.minimized
            cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor
            drag.target: root
            drag.axis: Drag.XAndYAxis
            drag.minimumX: 0
            drag.maximumX: Math.max(0, root.parent.width - root.width)
            drag.minimumY: 0
            drag.maximumY: Math.max(0, root.parent.height - root.height)
            onReleased: root.panelModel.setPosition(root.x, root.y)
        }

        MouseArea {
            id: collapseArea
            anchors.right: parent.right
            anchors.rightMargin: 2
            width: 22
            height: parent.height
            cursorShape: Qt.PointingHandCursor
            onClicked: root.minimized = !root.minimized
        }
    }

    ColumnLayout {
        id: body
        visible: !root.minimized
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: header.bottom
        anchors.bottom: parent.bottom
        anchors.margins: 10
        spacing: 5
    }
}
