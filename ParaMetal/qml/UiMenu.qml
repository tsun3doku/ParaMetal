import QtQuick
import QtQuick.Controls

Menu {
    id: control

    required property QtObject theme
    property real tabWidth: 0
    property Item anchorItem: null

    readonly property real contactTop: (anchorItem && anchorItem.visible && visible && menuBackground)
        ? Math.round(anchorItem.mapToItem(menuBackground, 0, 0).y)
        : -1
    readonly property real contactBottom: (anchorItem && anchorItem.visible && visible && contactTop >= 0)
        ? Math.round(contactTop + anchorItem.height)
        : -1

    popupType: Popup.Item
    overlap: 1
    padding: 1
    topPadding: 1
    bottomPadding: 1
    leftPadding: 1
    rightPadding: 1
    font.family: theme.fontFamily
    font.pixelSize: theme.regularFontSize
    font.weight: Font.Light

    delegate: UiMenuItem {
        theme: control.theme
    }

    background: Rectangle {
        id: menuBackground
        implicitWidth: 120
        color: control.theme.menuBarBackground
        radius: 0

        // Left border - top segment (above contact region, if any)
        Rectangle {
            anchors.left: parent.left
            anchors.top: parent.top
            height: (control.anchorItem && control.contactTop >= 0) ? control.contactTop : parent.height
            width: 1
            color: control.theme.border
            visible: height > 0
        }

        // Left border - bottom segment (below contact region)
        Rectangle {
            anchors.left: parent.left
            anchors.bottom: parent.bottom
            height: (control.anchorItem && control.contactBottom >= 0)
                    ? Math.max(0, parent.height - control.contactBottom)
                    : 0
            width: 1
            color: control.theme.border
            visible: height > 0 && control.anchorItem !== null
        }

        // Seamless bridge into the parent menu item to erase the separating border line
        Rectangle {
            visible: control.anchorItem !== null && control.contactTop >= 0
            x: -2
            y: control.contactTop
            width: 3
            height: Math.max(0, control.contactBottom - control.contactTop)
            color: control.theme.hover
            z: 2
        }

        // Right border
        Rectangle {
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 1
            color: control.theme.border
        }

        // Bottom border
        Rectangle {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            height: 1
            color: control.theme.border
        }

        // Top border: if tabWidth > 0, omit the border directly beneath the active tab
        Rectangle {
            anchors.top: parent.top
            x: control.tabWidth > 0 ? control.tabWidth - 1 : 0
            width: control.tabWidth > 0 ? Math.max(0, parent.width - (control.tabWidth - 1)) : parent.width
            height: 1
            color: control.theme.border
            visible: width > 0
        }
    }
}
