import QtQuick
import QtQuick.Controls

MenuSeparator {
    required property QtObject theme

    leftPadding: 8
    rightPadding: 8
    topPadding: 3
    bottomPadding: 3

    contentItem: Rectangle {
        implicitHeight: 1
        color: theme.menuSeparator
    }
}
