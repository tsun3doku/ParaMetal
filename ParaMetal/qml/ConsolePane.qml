import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject bridge
    color: theme.panelBackground

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 32
            color: theme.panelBackground
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                spacing: 7
                Image {
                    source: "../textures/icons/Terminal/32w/Artboard 1.png"
                    sourceSize.width: 18
                    sourceSize.height: 18
                }
                Text {
                    text: qsTr("Console")
                    color: theme.text
                    font.family: theme.fontFamily
                    font.pixelSize: theme.titleFontSize
                    font.letterSpacing: 0.75
                }
                Item { Layout.fillWidth: true }
            }
        }

        ScrollView {
            id: outputScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            TextArea {
                id: output
                text: bridge.output
                readOnly: true
                selectByMouse: true
                wrapMode: TextEdit.Wrap
                color: theme.text
                font.family: theme.monoFamily
                font.pixelSize: theme.consoleFontSize
                leftPadding: 10
                rightPadding: 10
                topPadding: 6
                bottomPadding: 6
                background: Rectangle { color: theme.panelBackground }
                onTextChanged: cursorPosition = length
            }
        }

        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: theme.subtleBorder }

        Rectangle {
            id: actionHost
            Layout.fillWidth: true
            Layout.preferredHeight: sampleCard.visible ? 172 : 0
            color: theme.panelBackground
            clip: true

            Rectangle {
                id: sampleCard
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 10
                anchors.topMargin: 14
                anchors.bottomMargin: 14
                radius: 6
                color: theme.cardBackground
                border.width: 1
                border.color: theme.border

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 12
                    anchors.rightMargin: 2
                    anchors.topMargin: 8
                    anchors.bottomMargin: 8
                    spacing: 14

                    Image {
                        Layout.preferredWidth: 128
                        Layout.preferredHeight: 128
                        source: "../textures/preview.png"
                        fillMode: Image.PreserveAspectCrop
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.topMargin: 10
                        Layout.rightMargin: 20
                        Layout.minimumWidth: 0
                        spacing: 3
                        Text { text: qsTr("Sample graph available"); color: theme.text; font.family: theme.fontFamily; font.pixelSize: theme.titleFontSize; font.letterSpacing: 0.75; wrapMode: Text.Wrap }
                        Text { text: qsTr("default_graph() creates a sample graph"); color: theme.mutedText; font.family: theme.fontFamily; font.pixelSize: theme.descriptionFontSize; wrapMode: Text.Wrap; Layout.fillWidth: true }
                        Item { Layout.fillHeight: true }
                        Button {
                            id: runButton
                            Layout.alignment: Qt.AlignRight
                            text: qsTr("Run")
                            font.family: theme.fontFamily
                            font.pixelSize: theme.regularFontSize
                            onClicked: {
                                bridge.resetDefaultGraph()
                                sampleCard.visible = false
                            }
                            implicitWidth: 56
                            background: Rectangle {
                                radius: 4
                                color: runButton.hovered ? "#4180ff" : "#3578ff"
                            }
                            contentItem: Text { text: runButton.text; color: "white"; font: runButton.font; horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter }
                        }
                    }

                    Item {
                        Layout.preferredWidth: 22
                        Layout.minimumWidth: 22
                        Layout.maximumWidth: 22
                        Layout.preferredHeight: 22
                        Layout.alignment: Qt.AlignTop
                        transform: Translate { y: -4 }
                        MouseArea {
                            id: closeArea
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: sampleCard.visible = false
                        }
                        Image {
                            anchors.centerIn: parent
                            source: "../textures/icons/Menu/x/32w/Artboard 1.png"
                            sourceSize.width: 10
                            sourceSize.height: 10
                            fillMode: Image.PreserveAspectFit
                            opacity: closeArea.containsMouse ? 1.0 : 0.75
                        }
                    }
                }
            }
        }

        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: theme.subtleBorder }

        TextField {
            id: input
            Layout.fillWidth: true
            Layout.preferredHeight: 34
            placeholderText: ">>>"
            color: theme.text
            placeholderTextColor: theme.mutedText
            font.family: theme.monoFamily
            font.pixelSize: theme.consoleFontSize
            leftPadding: 10
            rightPadding: 10
            background: Rectangle { color: theme.panelBackground }
            onAccepted: {
                bridge.execute(text)
                text = ""
            }
        }
    }
}
