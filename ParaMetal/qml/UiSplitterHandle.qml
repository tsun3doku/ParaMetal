import QtQuick

Rectangle {
    id: control

    property int orientation: Qt.Horizontal

    implicitWidth: orientation === Qt.Horizontal ? UiTheme.splitterWidth : 0
    implicitHeight: orientation === Qt.Vertical ? UiTheme.splitterWidth : 0
    color: UiTheme.windowBackground
}
