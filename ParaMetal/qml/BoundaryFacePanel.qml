pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts

FloatingPanel {
    id: root
    required property QtObject boundaryFace
    required property QtObject graphModel

    title: qsTr("Boundary")
    panelModel: boundaryFace
    panelWidth: 260
    expandedHeight: contentColumn.implicitHeight + headerHeight + 20

    Column {
        id: contentColumn
        Layout.fillWidth: true
        spacing: 5

        BoundaryFaceFields {
            width: parent.width
            theme: root.theme
            graphModel: root.graphModel
            base: 100 + root.boundaryFace.face * 10
        }
    }
}
