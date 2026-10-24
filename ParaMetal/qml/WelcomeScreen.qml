import QtQuick
import QtQuick.Controls
import QtQuick.Effects
import QtQuick.Layouts

Item {
    id: root

    required property QtObject theme

    readonly property int choicePreviewSize: 152
    readonly property int choiceContentSpacing: 10
    readonly property int choiceTitleHeight: 18
    readonly property int choiceTileHeight: choicePreviewSize + choiceContentSpacing + choiceTitleHeight

    signal emptyGraphRequested()
    signal defaultGraphRequested()
    signal openProjectRequested()

    focus: visible
    Keys.onEscapePressed: root.visible = false

    Rectangle {
        anchors.fill: parent
        color: root.theme.welcomeBackdrop

        MouseArea {
            anchors.fill: parent
            onClicked: root.visible = false
        }
    }

    component ChoiceButton: Button {
        id: choice

        required property string titleText
        property bool emphasized: false

        Layout.preferredWidth: root.choicePreviewSize
        Layout.minimumWidth: root.choicePreviewSize
        Layout.maximumWidth: root.choicePreviewSize
        Layout.preferredHeight: root.choiceTileHeight
        Layout.minimumHeight: root.choiceTileHeight
        Layout.maximumHeight: root.choiceTileHeight
        padding: 0
        hoverEnabled: true

        contentItem: ColumnLayout {
            anchors.fill: parent
            spacing: root.choiceContentSpacing

            Item {
                Layout.preferredWidth: root.choicePreviewSize
                Layout.minimumWidth: root.choicePreviewSize
                Layout.maximumWidth: root.choicePreviewSize
                Layout.preferredHeight: root.choicePreviewSize
                Layout.minimumHeight: root.choicePreviewSize
                Layout.maximumHeight: root.choicePreviewSize
                Layout.alignment: Qt.AlignHCenter
                clip: true

                Image {
                    anchors.fill: parent
                    anchors.margins: 16
                    source: "../textures/preview.png"
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    mipmap: true
                    visible: choice.emphasized
                }

                Image {
                    anchors.fill: parent
                    anchors.margins: 12
                    source: "../textures/grid.png"
                    fillMode: Image.PreserveAspectFit
                    smooth: true
                    mipmap: true
                    visible: !choice.emphasized
                }
            }

            Text {
                Layout.fillWidth: true
                Layout.preferredHeight: root.choiceTitleHeight
                text: choice.titleText
                color: choice.hovered ? root.theme.welcomeText : root.theme.welcomeSubtext
                font.family: root.theme.fontFamily
                font.pixelSize: 13
                font.weight: Font.Medium
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
        }

        background: Rectangle {
            width: parent.width
            height: root.choicePreviewSize
            radius: 2
            color: choice.hovered ? root.theme.welcomeOptionHover : root.theme.welcomeOptionBackground
            border.width: choice.hovered ? 2 : 1
            border.color: choice.hovered ? root.theme.accent : root.theme.welcomeCardBorder
        }
    }

    Rectangle {
        id: welcomeCard
        anchors.centerIn: parent
        width: Math.min(820, parent.width - 60)
        height: Math.min(520, parent.height - 60)
        radius: 0
        color: root.theme.welcomeCardBackground
        clip: true

        MouseArea {
            anchors.fill: parent
            // Intercept clicks on the card background so they do not dismiss the splash screen
        }

        Image {
            id: splashImage
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: Math.round(parent.width * 0.48)
            source: "../textures/burning_man.jpg"
            fillMode: Image.PreserveAspectFit
            horizontalAlignment: Image.AlignLeft
            verticalAlignment: Image.AlignBottom
            smooth: true
            mipmap: true

            onStatusChanged: {
                if (status === Image.Error && source == "../textures/burning_man.jpg") {
                    source = "../../textures/burning_man.jpg"
                }
            }
        }

        Image {
            id: logoImage
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.leftMargin: 24
            anchors.topMargin: 24
            width: 42
            height: 42
            source: "../resources/Logo/app_icon.png"
            sourceSize.width: 84
            sourceSize.height: 84
            fillMode: Image.PreserveAspectFit
            smooth: true
            mipmap: true

            onStatusChanged: {
                if (status === Image.Error) {
                    if (source == "../resources/Logo/app_icon.png") {
                        source = "../../../resources/Logo/app_icon.png"
                    } else if (source == "../../../resources/Logo/app_icon.png") {
                        source = "../../resources/Logo/app_icon.png"
                    }
                }
            }
        }

        Text {
            id: artistCredit
            x: Math.round((splashImage.paintedWidth > 0 ? splashImage.paintedWidth : splashImage.width) - width - 20)
            anchors.bottom: splashImage.bottom
            anchors.bottomMargin: 14
            text: qsTr("Art by Daniel Isles")
            color: root.theme.welcomeSubtext
            font.family: root.theme.fontFamily
            font.pixelSize: root.theme.consoleFontSize
            font.weight: Font.Normal
            opacity: 0.7
        }

        Text {
            id: versionText
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.rightMargin: 24
            anchors.topMargin: 20
            text: "v1.7.0"
            color: root.theme.welcomeSubtext
            font.family: root.theme.fontFamily
            font.pixelSize: root.theme.consoleFontSize
            font.weight: Font.Normal
            opacity: 0.7
        }

        Rectangle {
            anchors.fill: parent
            radius: 0
            color: "transparent"
            border.width: 1
            border.color: root.theme.welcomeCardBorder
        }

        ColumnLayout {
            anchors.right: parent.right
            anchors.rightMargin: 68
            anchors.verticalCenter: parent.verticalCenter
            spacing: 20

            ColumnLayout {
                Layout.alignment: Qt.AlignLeft
                spacing: 6

                Text {
                    text: qsTr("Start a Project")
                    color: root.theme.welcomeText
                    font.family: root.theme.fontFamily
                    font.pixelSize: 26
                    font.weight: Font.DemiBold
                    horizontalAlignment: Text.AlignLeft
                }

                Text {
                    text: qsTr("Choose a starting point for your workspace")
                    color: root.theme.welcomeSubtext
                    font.family: root.theme.fontFamily
                    font.pixelSize: 13
                    font.weight: Font.Normal
                    horizontalAlignment: Text.AlignLeft
                }
            }

            Button {
                id: openProjectButton
                Layout.alignment: Qt.AlignLeft
                Layout.preferredHeight: 28
                padding: 0
                hoverEnabled: true
                onClicked: root.openProjectRequested()

                contentItem: RowLayout {
                    spacing: 8

                    Image {
                        Layout.preferredWidth: 18
                        Layout.preferredHeight: 18
                        Layout.alignment: Qt.AlignVCenter
                        source: "../textures/icons/File/open/32w/Artboard 1.png"
                        sourceSize.width: 36
                        sourceSize.height: 36
                        fillMode: Image.PreserveAspectFit
                        layer.enabled: true
                        layer.effect: MultiEffect {
                            colorization: openProjectButton.hovered || openProjectButton.pressed ? 1.0 : 0.0
                            colorizationColor: root.theme.accent
                        }
                        opacity: openProjectButton.hovered || openProjectButton.pressed ? 1.0 : 0.85
                    }

                    Text {
                        Layout.alignment: Qt.AlignVCenter
                        text: qsTr("Open Project")
                        color: openProjectButton.hovered || openProjectButton.pressed
                               ? root.theme.accent : root.theme.welcomeText
                        font.family: root.theme.fontFamily
                        font.pixelSize: 13
                        font.weight: Font.Medium
                    }
                }

                background: Item {}
            }

            RowLayout {
                Layout.alignment: Qt.AlignLeft
                spacing: 20

                ChoiceButton {
                    titleText: qsTr("Empty Graph")
                    onClicked: root.emptyGraphRequested()
                }

                ChoiceButton {
                    titleText: qsTr("Default Graph")
                    emphasized: true
                    onClicked: root.defaultGraphRequested()
                }
            }
        }
    }
}
