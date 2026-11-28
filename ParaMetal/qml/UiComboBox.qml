pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls

ComboBox {
    id: control
    required property QtObject theme
    property string placeholderText: ""
    property bool searchable: false
    property string fallbackText: ""
    signal textCommitted(string text)

    readonly property string currentDisplayText: displayText.length > 0 ? displayText : fallbackText
    property int searchMatchIndex: -1
    property int focusStartIndex: -1
    property bool itemPressInProgress: false

    implicitHeight: 28
    font: theme.regularFont
    editable: searchable

    hoverEnabled: true

    function matchIndexFor(value) {
        const needle = String(value).toLocaleLowerCase()
        if (needle.length === 0) return -1
        for (let index = 0; index < control.count; ++index) {
            if (control.textAt(index).toLocaleLowerCase().indexOf(needle) >= 0)
                return index
        }
        return -1
    }

    function commitText(value) {
        const text = String(value)
        control.popup.close()
        control.searchMatchIndex = -1
        if (text === control.currentDisplayText)
            return
        control.textCommitted(text)
    }

    onActivated: {
        if (!control.searchable) return
        control.editText = control.currentText
        control.focusStartIndex = control.currentIndex
        control.searchMatchIndex = -1
    }

    onCountChanged: control.searchMatchIndex = -1

    contentItem: Loader {
        sourceComponent: control.searchable ? editableContent : displayContent
    }

    Component {
        id: displayContent
        Text {
            leftPadding: 7
            rightPadding: 20
            text: control.currentDisplayText.length > 0
                ? control.currentDisplayText
                : control.placeholderText
            color: control.theme.text
            font: control.font
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    Component {
        id: editableContent
        TextField {
            id: editField
            leftPadding: 7
            rightPadding: 20
            topPadding: 0
            bottomPadding: 0
            text: control.activeFocus ? control.editText : control.currentDisplayText
            color: control.theme.text
            selectionColor: control.theme.accent
            selectedTextColor: control.theme.onAccentText
            placeholderText: control.placeholderText
            placeholderTextColor: control.theme.mutedText
            font: control.font
            verticalAlignment: TextInput.AlignVCenter
            background: null
            selectByMouse: true

            onActiveFocusChanged: {
                if (activeFocus) {
                    control.editText = control.currentDisplayText
                    control.focusStartIndex = control.currentIndex
                    control.searchMatchIndex = control.matchIndexFor(control.editText)
                    selectAll()
                } else if (control.searchable) {
                    if (!control.itemPressInProgress &&
                        control.currentIndex === control.focusStartIndex) {
                        control.commitText(control.editText)
                    }
                    control.searchMatchIndex = -1
                }
            }

            onTextEdited: {
                control.editText = text
                control.searchMatchIndex = control.matchIndexFor(text)
                if (!control.popup.visible)
                    control.popup.open()
            }

            onAccepted: control.commitText(text)
        }
    }

    indicator: Image {
        x: control.width - width - 8
        y: (control.height - height) / 2
        width: 10
        height: 10
        source: "../textures/icons/Overlays/arrow/128w/Artboard 1.png"
        sourceSize.width: width
        sourceSize.height: height
        fillMode: Image.PreserveAspectFit
        opacity: control.hovered ? 1.0 : 0.8

        MouseArea {
            anchors.fill: parent
            enabled: control.searchable
            visible: enabled
            cursorShape: Qt.PointingHandCursor
            onClicked: {
                control.searchMatchIndex = -1
                if (control.popup.visible) control.popup.close()
                else control.popup.open()
            }
        }
    }

    background: Rectangle {
        radius: 3
        color: control.down ? control.theme.selected
             : (control.hovered || control.popup.visible) ? control.theme.toolHover
             : control.theme.inputBackground
        border.width: 1
        border.color: (control.activeFocus || control.popup.visible) ? control.theme.accent : control.theme.border
    }

    popup: Popup {
        y: control.height - 1
        width: control.width
        padding: 1

        onOpened: control.itemPressInProgress = false
        onClosed: {
            control.searchMatchIndex = -1
            Qt.callLater(function() {
                control.itemPressInProgress = false
            })
        }

        contentItem: ListView {
            id: popupList
            clip: true
            implicitHeight: Math.min(contentHeight, 260)
            model: control.delegateModel
            currentIndex: control.searchable ? control.searchMatchIndex : control.highlightedIndex
            highlightMoveDuration: 0

            onCurrentIndexChanged: {
                if (control.searchable && currentIndex >= 0)
                    positionViewAtIndex(currentIndex, ListView.Contain)
            }

            delegate: ItemDelegate {
                id: delegateItem
                required property int index
                hoverEnabled: true
                width: control.width - 2
                height: 24
                padding: 0
                highlighted: control.searchable
                    ? control.searchMatchIndex === index
                    : control.highlightedIndex === index
                onHoveredChanged: {
                    if (hovered && control.searchable)
                        control.searchMatchIndex = index
                }
                onPressed: control.itemPressInProgress = true
                contentItem: Text {
                    leftPadding: 10
                    text: control.textAt(delegateItem.index)
                    color: (delegateItem.highlighted || delegateItem.hovered)
                           ? control.theme.headingText
                           : control.theme.text
                    font: control.theme.regularFont
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    radius: 0
                    color: delegateItem.down ? control.theme.selected
                         : (delegateItem.highlighted || delegateItem.hovered) ? control.theme.hover
                         : "transparent"
                }
            }
        }
        background: Rectangle {
            color: control.theme.popupBackground
            border.width: 1
            border.color: control.theme.border
            radius: 3
        }
    }
}
