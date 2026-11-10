import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject bridge
    color: theme.panelBackground
    border.width: 1
    border.color: theme.subtleBorder

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 1
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
                    source: "../textures/icons/Terminal/128w/Artboard 1.png"
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
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            TextArea {
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

        TextField {
            id: inputField
            Layout.fillWidth: true
            Layout.preferredHeight: 42
            verticalAlignment: TextInput.AlignVCenter
            placeholderText: ">>>"
            color: theme.text
            placeholderTextColor: theme.mutedText
            font.family: theme.monoFamily
            font.pixelSize: theme.consoleFontSize
            leftPadding: 10
            rightPadding: 10
            background: Rectangle { color: theme.panelBackground }

            property var commandHistory: []
            property int historyIndex: 0
            property string draftInput: ""

            onAccepted: {
                var cmd = text
                if (cmd.length > 0) {
                    commandHistory.push(cmd)
                    historyIndex = commandHistory.length
                }
                draftInput = ""
                bridge.execute(cmd)
                text = ""
            }

            Keys.onUpPressed: function(event) {
                if (commandHistory.length === 0) return
                if (historyIndex === commandHistory.length) {
                    draftInput = text
                }
                if (historyIndex > 0) {
                    historyIndex--
                    text = commandHistory[historyIndex]
                    cursorPosition = text.length
                }
                event.accepted = true
            }

            Keys.onDownPressed: function(event) {
                if (commandHistory.length === 0) return
                if (historyIndex < commandHistory.length - 1) {
                    historyIndex++
                    text = commandHistory[historyIndex]
                    cursorPosition = text.length
                } else if (historyIndex === commandHistory.length - 1) {
                    historyIndex = commandHistory.length
                    text = draftInput
                    cursorPosition = text.length
                }
                event.accepted = true
            }

            Keys.onTabPressed: function(event) {
                var pos = cursorPosition
                insert(pos, "    ")
                cursorPosition = pos + 4
                event.accepted = true
            }

            Keys.onEscapePressed: function(event) {
                text = ""
                draftInput = ""
                historyIndex = commandHistory.length
                event.accepted = true
            }
        }
    }
}
