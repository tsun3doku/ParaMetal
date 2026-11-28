import QtQuick

Item {
    id: root

    required property QtObject theme
    required property QtObject bridge

    property string viewportMode: "Display"

    implicitWidth: UiTheme.toolButtonGroupWidth(3)
    implicitHeight: column.implicitHeight

    function selectElement(elementType) {
        if (!root.bridge.modelSelected && !root.bridge.meshSelectionActive)
            return
        root.bridge.setMeshSelectionElementType(elementType)
        if (!root.bridge.meshSelectionActive)
            root.bridge.setMeshSelectionActive(true)
    }

    Shortcut {
        sequence: "1"
        enabled: root.bridge.meshSelectionActive
        onActivated: root.selectElement(0)
    }

    Shortcut {
        sequence: "2"
        enabled: root.bridge.meshSelectionActive
        onActivated: root.selectElement(1)
    }

    Shortcut {
        sequence: "3"
        enabled: root.bridge.meshSelectionActive
        onActivated: root.selectElement(2)
    }

    Column {
        id: column
        width: parent.width
        spacing: 6

        UiComboBox {
            width: parent.width
            height: UiTheme.toolButtonSize + UiTheme.toolButtonFrameSize
            theme: root.theme
            model: [qsTr("View"), qsTr("Select")]
            currentIndex: root.viewportMode === "Select" ? 1 : 0
            onActivated: index => {
                if (index === 1) {
                    root.viewportMode = "Select"
                } else {
                    root.viewportMode = "Display"
                    root.bridge.setMeshSelectionActive(false)
                }
            }
        }

        Rectangle {
            x: (parent.width - width) / 2
            width: UiTheme.toolButtonGroupWidth(3)
            height: UiTheme.toolButtonSize + UiTheme.toolButtonFrameSize
            radius: 5
            color: root.theme.toolBorder
            visible: root.viewportMode === "Select"

            Row {
                x: UiTheme.toolButtonFrameInset
                y: UiTheme.toolButtonFrameInset
                spacing: UiTheme.toolButtonGroupSpacing

                ToolIconButton {
                    segment: 1
                    iconSize: 20
                    checkable: false
                    preserveSourceColorWhenChecked: true
                    inactiveIconColor: "#808080"
                    baseColor: root.theme.toolBorder
                    enabled: root.bridge.modelSelected || root.bridge.meshSelectionActive
                    imageSource: "../textures/icons/Overlays/subselection/vertex/128w/Artboard 1.png"
                    checked: root.bridge.meshSelectionActive &&
                        root.bridge.meshSelectionElementType === 0
                    onClicked: root.selectElement(0)
                }

                ToolIconButton {
                    segment: 2
                    iconSize: 20
                    checkable: false
                    preserveSourceColorWhenChecked: true
                    inactiveIconColor: "#808080"
                    baseColor: root.theme.toolBorder
                    enabled: root.bridge.modelSelected || root.bridge.meshSelectionActive
                    imageSource: "../textures/icons/Overlays/subselection/edge/128w/Artboard 1.png"
                    checked: root.bridge.meshSelectionActive &&
                        root.bridge.meshSelectionElementType === 1
                    onClicked: root.selectElement(1)
                }

                ToolIconButton {
                    segment: 3
                    iconSize: 20
                    checkable: false
                    preserveSourceColorWhenChecked: true
                    inactiveIconColor: "#808080"
                    baseColor: root.theme.toolBorder
                    enabled: root.bridge.modelSelected || root.bridge.meshSelectionActive
                    imageSource: "../textures/icons/Overlays/subselection/face/128w/Artboard 1.png"
                    checked: root.bridge.meshSelectionActive &&
                        root.bridge.meshSelectionElementType === 2
                    onClicked: root.selectElement(2)
                }
            }
        }
    }
}
