import QtQuick
import QtQuick.Controls

Menu {
    id: control

    required property QtObject theme

    popupType: Popup.Item
    topPadding: 3
    bottomPadding: 3
    font.family: theme.fontFamily
    font.pixelSize: theme.regularFontSize
    font.weight: Font.Light

    delegate: UiMenuItem {
        theme: control.theme
    }

    background: Rectangle {
        implicitWidth: 110
        color: control.theme.panelBackground
        border.width: 1
        border.color: control.theme.toolBorder
    }
}
