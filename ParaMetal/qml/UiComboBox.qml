import QtQuick
import QtQuick.Controls

ComboBox {
    id: control
    required property QtObject theme
    property string placeholderText: ""

    implicitHeight: 28
    font: theme.regularFont

    hoverEnabled: true

    contentItem: Text {
        leftPadding: 7
        rightPadding: 20
        text: control.displayText.length > 0 ? control.displayText : control.placeholderText
        color: control.theme.text
        font: control.font
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    indicator: Text {
        x: control.width - width - 8
        y: (control.height - height) / 2
        text: "▾"
        color: control.hovered ? control.theme.headingText : control.theme.text
        font: control.theme.regularFont
    }

    background: Rectangle {
        radius: 0
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
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: control.delegateModel
            currentIndex: control.highlightedIndex
            delegate: ItemDelegate {
                id: delegateItem
                required property int index
                hoverEnabled: true
                width: control.width - 2
                height: 24
                padding: 0
                highlighted: control.highlightedIndex === index
                contentItem: Text {
                    leftPadding: 10
                    text: control.textAt(index)
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
            color: control.theme.cardBackground
            border.width: 1
            border.color: control.theme.border
            radius: 0
        }
    }
}
