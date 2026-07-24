import QtQuick
import QtQuick.Controls

Rectangle {
    id: root
    required property QtObject theme
    required property QtObject bridge
    color: "#1e1e1e"

    readonly property int marginX: 18
    readonly property int buttonSize: 20
    readonly property int buttonGap: 1
    readonly property int controlsWidth: buttonSize * 6 + buttonGap * 5
    readonly property int trackGap: 26
    readonly property int trackLeft: marginX + controlsWidth + trackGap
    readonly property int rightControls: 256
    readonly property int trackWidth: Math.max(1, width - trackLeft - rightControls - marginX)
    readonly property int centerY: Math.round(height / 2)
    readonly property int scrubHitPadding: 8
    readonly property int maximumFrame: Math.max(1, bridge.endFrame)
    readonly property real normalizedFrame: Math.max(0, Math.min(1, bridge.currentFrame / maximumFrame))
    readonly property int availableFrame: bridge.recordedFrames > 0
                                          ? Math.min(bridge.recordedFrames - 1, maximumFrame)
                                          : 0

    function playbackIcon(folder) {
        return "../textures/icons/Playback/" + folder + "/32w/Artboard 1.png"
    }

    Rectangle { anchors.left: parent.left; anchors.right: parent.right; height: 1; color: theme.border }

    Row {
        x: root.marginX
        y: Math.round((root.height - root.buttonSize) / 2)
        spacing: root.buttonGap

        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 7; segment: 1; checkable: false
            iconSize: 12
            imageSource: root.playbackIcon("Last_frame")
            mirrorHorizontal: true
            onClicked: bridge.scrub(0)
        }
        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 7; segment: 2; checkable: false
            iconSize: 12
            imageSource: root.playbackIcon("Next_frame")
            mirrorHorizontal: true
            onClicked: bridge.step(-1)
        }
        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 6; segment: 2
            iconSize: 14
            checked: bridge.playing
            imageSource: root.playbackIcon(bridge.playing ? "Pause" : "Start")
            onClicked: bridge.setPlaying(checked)
        }
        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 7; segment: 2; checkable: false
            iconSize: 12
            imageSource: root.playbackIcon("Next_frame")
            onClicked: bridge.step(1)
        }
        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 7; segment: 2; checkable: false
            iconSize: 12
            imageSource: root.playbackIcon("Last_frame")
            onClicked: bridge.scrub(root.maximumFrame)
        }
        ToolIconButton {
            width: root.buttonSize; height: root.buttonSize; padding: 6; segment: 3; checkable: false
            iconSize: 16
            imageSource: root.playbackIcon("Reset")
            onClicked: bridge.reset()
        }
    }

    Item {
        id: track
        x: root.trackLeft
        width: root.trackWidth
        anchors.top: parent.top
        anchors.bottom: parent.bottom

        Rectangle {
            y: root.centerY - 1
            width: parent.width
            height: 3
            color: "#41434c"
        }

        Rectangle {
            y: root.centerY - 1
            width: root.availableFrame > 0
                   ? Math.max(1, Math.round(parent.width * root.availableFrame / root.maximumFrame))
                   : 0
            height: 3
            color: "#416d9c"
            visible: root.availableFrame > 0
        }

        Repeater {
            model: 51
            delegate: Rectangle {
                required property int index
                readonly property bool major: index % 5 === 0
                x: Math.round(index * (track.width - 1) / 50)
                y: root.centerY - (major ? 7 : 6)
                width: major ? 2 : 1
                height: major ? 15 : 12
                color: "#767780"
            }
        }

        Repeater {
            model: 11
            delegate: Text {
                required property int index
                readonly property int frameNumber: Math.round(index * root.maximumFrame / 10)
                x: Math.round(index * track.width / 10) - 28
                y: root.centerY - 20
                width: 56
                height: 14
                text: frameNumber
                color: "#cccccc"
                font: root.theme.regularFont
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }

        Rectangle {
            x: Math.round(root.normalizedFrame * track.width) - 1
            y: root.centerY - 7
            width: 2
            height: 23
            color: "#2e7eff"
        }

        Rectangle {
            width: Math.max(32, frameLabel.implicitWidth + 12)
            height: 14
            x: Math.max(-width / 2, Math.min(track.width - width / 2, Math.round(root.normalizedFrame * track.width) - width / 2))
            y: root.centerY - 20
            radius: 6
            color: "#3578ff"
            Text {
                id: frameLabel
                anchors.centerIn: parent
                text: root.bridge.currentFrame
                color: "#f5f8ff"
                font: root.theme.regularFont
            }
        }

        MouseArea {
            x: -root.scrubHitPadding
            width: parent.width + root.scrubHitPadding * 2
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            function scrub(mouseX) {
                const trackX = mouseX - root.scrubHitPadding
                bridge.scrub(Math.round(Math.max(0, Math.min(1, trackX / track.width)) * root.maximumFrame))
            }
            onPressed: function(mouse) { scrub(mouse.x) }
            onPositionChanged: function(mouse) { if (pressed) scrub(mouse.x) }
        }
    }

    Item {
        x: root.trackLeft + root.trackWidth + 18
        width: root.rightControls - 18
        height: parent.height

        Text {
            x: 0; y: 5; width: 34; height: 14
            text: qsTr("Start")
            color: "#cccccc"
            font: root.theme.regularFont
            horizontalAlignment: Text.AlignRight; verticalAlignment: Text.AlignVCenter
        }
        Text {
            x: 0; y: 23; width: 34; height: 14
            text: qsTr("End")
            color: "#cccccc"
            font: root.theme.regularFont
            horizontalAlignment: Text.AlignRight; verticalAlignment: Text.AlignVCenter
        }
        Rectangle {
            x: 42; y: 5; width: 42; height: 14; radius: 5
            color: "#222328"; border.width: 1; border.color: "#3a3c44"
            Text { anchors.centerIn: parent; text: "0"; color: "#cccccc"; font: root.theme.regularFont }
        }
        Rectangle {
            x: 42; y: 23; width: 42; height: 14; radius: 5
            color: "#222328"; border.width: 1; border.color: "#3a3c44"
            Text { anchors.centerIn: parent; text: root.maximumFrame; color: "#cccccc"; font: root.theme.regularFont }
        }
        Text {
            x: 102; y: 14; height: 16
            text: bridge.currentSeconds.toFixed(2) + " / " + bridge.durationSeconds.toFixed(2) + " s"
            color: "#cccccc"
            font: root.theme.regularFont
            verticalAlignment: Text.AlignVCenter
        }
    }
}
