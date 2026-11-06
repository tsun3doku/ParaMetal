import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import ParaMetal

Rectangle {
    id: root
    width: 1600
    height: 900
    readonly property QtObject theme: UiTheme
    color: theme.windowBackground

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        MenuBar {
            id: mainMenuBar
            z: 100
            Layout.fillWidth: true
            Layout.preferredHeight: 25
            font.family: theme.fontFamily
            palette.window: theme.menuBarBackground
            palette.windowText: theme.text
            palette.buttonText: theme.text
            palette.text: theme.text
            palette.highlight: theme.hover
            palette.highlightedText: theme.headingText
            background: Rectangle {
                color: theme.menuBarBackground
            }
            delegate: MenuBarItem {
                id: menuBarItem
                readonly property bool isOpen: menuBarItem.menu && menuBarItem.menu.visible
                implicitWidth: contentItem.implicitWidth + 20
                implicitHeight: 25
                leftPadding: 10
                rightPadding: 10
                topPadding: 0
                bottomPadding: 0

                Binding {
                    target: menuBarItem.menu
                    property: "tabWidth"
                    value: menuBarItem.isOpen ? menuBarItem.width : 0
                    when: menuBarItem.menu !== null
                }

                contentItem: Text {
                    text: menuBarItem.text
                    color: (menuBarItem.highlighted || menuBarItem.isOpen) ? theme.headingText : theme.text
                    font.family: theme.fontFamily
                    font.pixelSize: theme.regularFontSize
                    font.weight: Font.Light
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                background: Rectangle {
                    color: menuBarItem.isOpen ? theme.menuBarBackground
                         : menuBarItem.highlighted ? theme.hover
                         : "transparent"

                    // Top border when open
                    Rectangle {
                        visible: menuBarItem.isOpen
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: 1
                        color: theme.border
                    }

                    // Left border when open - reaches down to y=25 to seamlessly meet UiMenu
                    Rectangle {
                        visible: menuBarItem.isOpen
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: -1
                        anchors.left: parent.left
                        width: 1
                        color: theme.border
                    }

                    // Right border when open - reaches down to y=25 to seamlessly meet UiMenu
                    Rectangle {
                        visible: menuBarItem.isOpen
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: -1
                        anchors.right: parent.right
                        width: 1
                        color: theme.border
                    }

                    // Bottom bridge when open to eliminate any separating line
                    Rectangle {
                        visible: menuBarItem.isOpen
                        anchors.left: parent.left
                        anchors.leftMargin: 1
                        anchors.right: parent.right
                        anchors.rightMargin: 1
                        anchors.bottom: parent.bottom
                        anchors.bottomMargin: -1
                        height: 2
                        color: theme.menuBarBackground
                    }
                }
            }

            UiMenu {
                y: mainMenuBar.height
                theme: root.theme
                title: qsTr("File")
                Action { text: qsTr("New"); onTriggered: welcomeScreen.visible = true }
                Action { text: qsTr("Open..."); onTriggered: openProjectDialog.open() }
                Action {
                    text: qsTr("Save")
                    onTriggered: {
                        if (project.path.length > 0) project.save()
                        else saveProjectDialog.open()
                    }
                }
                Action { text: qsTr("Save As..."); onTriggered: saveProjectDialog.open() }
                UiMenuSeparator { theme: root.theme }
                Action { text: qsTr("Exit"); onTriggered: Qt.quit() }
            }
            UiMenu {
                y: mainMenuBar.height
                theme: root.theme
                title: qsTr("View")
                Action { text: qsTr("Console"); checkable: true; checked: consolePane.visible; onTriggered: consolePane.visible = checked }
                Action { text: qsTr("Node Graph"); checkable: true; checked: nodeEditor.visible; onTriggered: nodeEditor.visible = checked }
            }
            UiMenu {
                y: mainMenuBar.height
                theme: root.theme
                title: qsTr("Project")
                Action { text: qsTr("Settings..."); onTriggered: projectSettings.open() }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            handle: UiSplitterHandle {
                orientation: Qt.Horizontal
            }

            ConsolePane {
                id: consolePane
                SplitView.preferredWidth: 250
                SplitView.minimumWidth: 0
                SplitView.maximumWidth: 520
                theme: root.theme
                bridge: ui.console
            }

            SplitView {
                SplitView.fillWidth: true
                SplitView.fillHeight: true
                orientation: Qt.Vertical

                handle: UiSplitterHandle {
                    orientation: Qt.Vertical
                }

                SplitView {
                    SplitView.fillWidth: true
                    SplitView.fillHeight: true
                    orientation: Qt.Horizontal

                    handle: UiSplitterHandle {
                        orientation: Qt.Horizontal
                    }

                    NodeEditorPane {
                        id: nodeEditor
                        SplitView.preferredWidth: 365
                        SplitView.minimumWidth: 365
                        SplitView.maximumWidth: 620
                        theme: root.theme
                        graphModel: ui.nodeGraph
                        runtimeNotifier: runtimeStatus
                    }

                    ViewportPane {
                        SplitView.fillWidth: true
                        SplitView.fillHeight: true
                        theme: root.theme
                        bridge: ui.viewport
                        heatPalette: ui.heatPalette
                    }
                }

                TimelineBar {
                    SplitView.fillWidth: true
                    SplitView.preferredHeight: 44
                    SplitView.minimumHeight: 44
                    SplitView.maximumHeight: 160
                    theme: root.theme
                    bridge: ui.timeline
                }
            }
        }
    }

    WelcomeScreen {
        id: welcomeScreen
        anchors.fill: parent
        z: 1000
        theme: root.theme
        onEmptyGraphRequested: {
            project.newEmptyProject()
            visible = false
        }
        onDefaultGraphRequested: {
            project.newProject()
            visible = false
        }
        onOpenProjectRequested: openProjectDialog.open()
    }

    FileDialog {
        id: openProjectDialog
        title: qsTr("Open ParaMetal Project")
        fileMode: FileDialog.OpenFile
        nameFilters: [qsTr("ParaMetal projects (*.pm)"), qsTr("All files (*)")]
        onAccepted: {
            welcomeScreen.visible = false
            project.open(selectedFile)
        }
    }

    FileDialog {
        id: saveProjectDialog
        title: qsTr("Save ParaMetal Project")
        fileMode: FileDialog.SaveFile
        defaultSuffix: "pm"
        nameFilters: [qsTr("ParaMetal projects (*.pm)"), qsTr("All files (*)")]
        onAccepted: project.saveAs(selectedFile)
    }

    MessageDialog {
        id: projectErrorDialog
        title: qsTr("Project Error")
        buttons: MessageDialog.Ok
    }

    Connections {
        target: project
        function onSaveAsRequired() { saveProjectDialog.open() }
        function onError(message) {
            projectErrorDialog.text = message
            projectErrorDialog.open()
        }
    }

    Popup {
        id: projectSettings
        modal: true
        anchors.centerIn: Overlay.overlay
        width: 300
        padding: 16
        background: Rectangle { color: theme.panelBackground; border.color: theme.subtleBorder }
        contentItem: ColumnLayout {
            spacing: 12
            Text { text: qsTr("Project Settings"); color: theme.headingText; font.pixelSize: theme.titleFontSize }
            RowLayout {
                Layout.fillWidth: true
                Text { text: qsTr("World Unit"); color: theme.text; font: theme.regularFont; Layout.fillWidth: true }
                UiComboBox {
                    theme: root.theme
                    Layout.preferredWidth: 150
                    Layout.preferredHeight: 28
                    model: [qsTr("Millimeters"), qsTr("Centimeters"), qsTr("Meters")]
                    currentIndex: project.worldUnit
                    onActivated: project.setWorldUnit(currentIndex)
                }
            }
            Button {
                id: closeButton
                text: qsTr("Close")
                Layout.alignment: Qt.AlignRight
                Layout.preferredHeight: 28
                onClicked: projectSettings.close()
                contentItem: Text {
                    text: closeButton.text
                    color: theme.text
                    font: theme.regularFont
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    implicitWidth: 84
                    radius: 3
                    color: closeButton.hovered ? theme.hover : theme.toolNormal
                    border.width: 1
                    border.color: theme.border
                }
            }
        }
    }

    Shortcut {
        sequence: "Escape"
        enabled: ui.nodeGraph.probePickingActive
        onActivated: ui.nodeGraph.cancelProbePicking()
    }
}
