pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Effects

Rectangle {
    id: root
    required property QtObject theme
    property real value: 0
    property var minimum: null
    property var maximum: null
    property bool hasRange: minimum !== null && maximum !== null
    property real step: 0
    property int decimals: 3
    property bool compact: false
    property bool readOnly: false
    signal valueEdited(var value)

    readonly property real rangeMin: minimum !== null ? Number(minimum) : -Infinity
    readonly property real rangeMax: maximum !== null ? Number(maximum) : Infinity

    readonly property bool canStepDown: {
        if (!hasRange || !isFinite(rangeMin)) return true
        return isFinite(root.value) && root.value > rangeMin
    }

    readonly property bool canStepUp: {
        if (!hasRange || !isFinite(rangeMax)) return true
        return isFinite(root.value) && root.value < rangeMax
    }

    HoverHandler {
        id: boxHover
    }

    readonly property bool isHovered: boxHover.hovered || scrubArea.containsMouse || leftArrowMouse.containsMouse || rightArrowMouse.containsMouse
    readonly property bool isModifying: numberField.activeFocus || scrubArea.dragging || leftArrowMouse.pressed || rightArrowMouse.pressed
    readonly property bool showArrows: isHovered || isModifying

    function formattedText() {
        if (!isFinite(root.value)) return ""
        if (root.decimals === 0) return String(Math.round(root.value))
        let formatted = root.value.toFixed(root.decimals)
        formatted = formatted.replace(/0+$/, "")
        if (formatted.endsWith(".")) formatted += "0"
        return formatted
    }

    function getStep() {
        if (root.step > 0) return root.step
        if (hasRange && isFinite(rangeMax) && isFinite(rangeMin)) {
            const span = rangeMax - rangeMin
            if (span > 5000 && root.decimals === 0) return Math.max(1, Math.round(span / 1000))
            if (span >= 50 && root.decimals <= 1) return 1
        }
        if (root.decimals === 0) return 1
        if (root.decimals === 1) return 0.1
        if (root.decimals === 2) return 0.01
        if (root.decimals === 3) return 0.001
        return Math.pow(10, -Math.min(root.decimals, 6))
    }

    function clampValue(val) {
        if (hasRange && isFinite(rangeMin) && isFinite(rangeMax)) {
            return Math.max(rangeMin, Math.min(rangeMax, val))
        }
        return val
    }

    function roundToDecimals(val) {
        if (root.decimals === 0) return Math.round(val)
        const factor = Math.pow(10, root.decimals)
        return Math.round(val * factor) / factor
    }

    function stepValue(direction, multiplier) {
        if (root.readOnly) return
        const current = Number(root.value)
        if (!isFinite(current)) return
        const mult = (multiplier !== undefined && multiplier > 0) ? multiplier : 1.0
        const stp = getStep() * mult
        const nextVal = roundToDecimals(clampValue(current + direction * stp))
        root.valueEdited(nextVal)
    }

    onValueChanged: {
        if (!numberField.activeFocus) {
            numberField.text = root.formattedText()
        }
    }

    implicitWidth: 120
    implicitHeight: 28
    radius: 3
    clip: true

    // Overall box background: highlights smoothly on hover
    color: numberField.activeFocus
           ? root.theme.inputBackground
           : (root.isHovered && !root.readOnly
              ? Qt.lighter(root.theme.inputBackground, 1.25)
              : root.theme.inputBackground)

    border.width: 1
    border.color: numberField.activeFocus
                  ? root.theme.accent
                  : (root.isHovered && !root.readOnly
                     ? Qt.lighter(root.theme.border, 1.35)
                     : root.theme.border)

    Behavior on color {
        ColorAnimation { duration: 50; easing.type: Easing.OutQuad }
    }
    Behavior on border.color {
        ColorAnimation { duration: 50; easing.type: Easing.OutQuad }
    }

    // Left Arrow Button (<)
    Item {
        id: leftArrow
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: 16
        z: 2
        enabled: !root.readOnly && root.canStepDown

        opacity: root.showArrows ? 1.0 : 0.0
        visible: opacity > 0
        Behavior on opacity {
            NumberAnimation { duration: 50; easing.type: Easing.OutQuad }
        }

        Image {
            id: leftArrowImg
            anchors.centerIn: parent
            width: 8
            height: 8
            source: "../textures/icons/Playback/Next_frame/32w/Artboard 1.png"
            sourceSize.width: 32
            sourceSize.height: 32
            smooth: true
            mipmap: true
            fillMode: Image.PreserveAspectFit
            mirror: true
            opacity: (!root.canStepDown || root.readOnly)
                     ? 0.2
                     : (leftArrowMouse.containsMouse ? 1.0 : (leftArrowMouse.pressed ? 1.0 : 0.65))
            layer.enabled: true
            layer.effect: MultiEffect {
                colorization: 1.0
                colorizationColor: leftArrowMouse.pressed ? root.theme.accent : root.theme.text
            }
        }

        MouseArea {
            id: leftArrowMouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: (!root.readOnly && root.canStepDown) ? Qt.PointingHandCursor : Qt.ArrowCursor
            preventStealing: true

            Timer {
                id: leftRepeatTimer
                interval: 60
                repeat: true
                onTriggered: {
                    if (root.canStepDown) root.stepValue(-1)
                    else stop()
                }
            }

            Timer {
                id: leftInitialTimer
                interval: 350
                onTriggered: {
                    if (leftArrowMouse.pressed) leftRepeatTimer.start()
                }
            }

            onPressed: mouse => {
                numberField.focus = false
                if (!root.readOnly && root.canStepDown) {
                    let stepMultiplier = 1.0
                    if (mouse.modifiers & Qt.ShiftModifier) stepMultiplier = 0.1
                    else if (mouse.modifiers & Qt.ControlModifier) stepMultiplier = 10.0
                    root.stepValue(-1, stepMultiplier)
                    leftInitialTimer.start()
                }
            }
            onReleased: {
                leftInitialTimer.stop()
                leftRepeatTimer.stop()
            }
            onCanceled: {
                leftInitialTimer.stop()
                leftRepeatTimer.stop()
            }
        }
    }

    // Right Arrow Button (>)
    Item {
        id: rightArrow
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: 16
        z: 2
        enabled: !root.readOnly && root.canStepUp

        opacity: root.showArrows ? 1.0 : 0.0
        visible: opacity > 0
        Behavior on opacity {
            NumberAnimation { duration: 50; easing.type: Easing.OutQuad }
        }

        Image {
            id: rightArrowImg
            anchors.centerIn: parent
            width: 8
            height: 8
            source: "../textures/icons/Playback/Next_frame/32w/Artboard 1.png"
            sourceSize.width: 32
            sourceSize.height: 32
            smooth: true
            mipmap: true
            fillMode: Image.PreserveAspectFit
            mirror: false
            opacity: (!root.canStepUp || root.readOnly)
                     ? 0.2
                     : (rightArrowMouse.containsMouse ? 1.0 : (rightArrowMouse.pressed ? 1.0 : 0.65))
            layer.enabled: true
            layer.effect: MultiEffect {
                colorization: 1.0
                colorizationColor: rightArrowMouse.pressed ? root.theme.accent : root.theme.text
            }
        }

        MouseArea {
            id: rightArrowMouse
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: (!root.readOnly && root.canStepUp) ? Qt.PointingHandCursor : Qt.ArrowCursor
            preventStealing: true

            Timer {
                id: rightRepeatTimer
                interval: 60
                repeat: true
                onTriggered: {
                    if (root.canStepUp) root.stepValue(1)
                    else stop()
                }
            }

            Timer {
                id: rightInitialTimer
                interval: 350
                onTriggered: {
                    if (rightArrowMouse.pressed) rightRepeatTimer.start()
                }
            }

            onPressed: mouse => {
                numberField.focus = false
                if (!root.readOnly && root.canStepUp) {
                    let stepMultiplier = 1.0
                    if (mouse.modifiers & Qt.ShiftModifier) stepMultiplier = 0.1
                    else if (mouse.modifiers & Qt.ControlModifier) stepMultiplier = 10.0
                    root.stepValue(1, stepMultiplier)
                    rightInitialTimer.start()
                }
            }
            onReleased: {
                rightInitialTimer.stop()
                rightRepeatTimer.stop()
            }
            onCanceled: {
                rightInitialTimer.stop()
                rightRepeatTimer.stop()
            }
        }
    }

    // Center Area: Centered Text Field + Brighter Modifiable Box Highlight + Scrubbing MouseArea
    Item {
        id: centerContainer
        anchors.left: leftArrow.right
        anchors.right: rightArrow.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        z: 1

        // Brighter on-hover highlight specifically for the modifiable box portion
        Rectangle {
            id: centerHighlight
            anchors.fill: parent
            anchors.margins: 1
            radius: 2
            visible: !numberField.activeFocus && !root.readOnly
            opacity: scrubArea.containsMouse ? 1.0 : 0.0
            color: Qt.lighter(root.theme.inputBackground, 1.45)

            Behavior on opacity {
                NumberAnimation { duration: 40; easing.type: Easing.OutQuad }
            }
        }

        TextField {
            id: numberField
            anchors.fill: parent
            horizontalAlignment: TextInput.AlignHCenter
            verticalAlignment: TextInput.AlignVCenter
            text: root.formattedText()
            color: root.theme.text
            selectionColor: root.theme.accent
            selectedTextColor: root.theme.onAccentText
            font: root.theme.regularFont
            readOnly: root.readOnly
            padding: 0
            leftPadding: root.theme.inspectorLabelSpacing
            rightPadding: root.theme.inspectorLabelSpacing
            topPadding: 0
            bottomPadding: 0
            background: null
            selectByMouse: true
            activeFocusOnPress: true

            function commit() {
                if (!activeFocus) return
                const raw = text.trim()
                const parsed = Number(raw)
                if (!isNaN(parsed) && raw.length > 0) {
                    const clamped = root.clampValue(parsed)
                    const rounded = root.roundToDecimals(clamped)
                    root.valueEdited(rounded)
                }
                focus = false
                text = Qt.binding(function() { return root.formattedText() })
            }

            onEditingFinished: commit()

            Keys.onReturnPressed: commit()
            Keys.onEnterPressed: commit()
            Keys.onEscapePressed: {
                focus = false
                text = Qt.binding(function() { return root.formattedText() })
            }

            onActiveFocusChanged: {
                if (!activeFocus) {
                    text = Qt.binding(function() { return root.formattedText() })
                }
            }
        }

        MouseArea {
            id: scrubArea
            anchors.fill: parent
            enabled: !numberField.activeFocus && !root.readOnly
            hoverEnabled: true
            cursorShape: dragging ? Qt.SizeHorCursor : (containsMouse ? Qt.SizeHorCursor : Qt.ArrowCursor)
            preventStealing: true

            property bool dragging: false
            property real startX: 0
            property real startVal: 0

            onPressed: mouse => {
                startX = mouse.x
                startVal = Number(root.value)
                if (!isFinite(startVal)) startVal = 0
                dragging = false
            }

            onPositionChanged: mouse => {
                if (!pressed) return
                const diff = mouse.x - startX
                if (!dragging && Math.abs(diff) >= 3) {
                    dragging = true
                }
                if (dragging) {
                    let sens = 1.0
                    if (root.hasRange && isFinite(root.rangeMax) && isFinite(root.rangeMin)) {
                        sens = (root.rangeMax - root.rangeMin) / Math.max(120, root.width * 2)
                    } else {
                        sens = root.getStep() * 0.5
                    }
                    if (mouse.modifiers & Qt.ShiftModifier) {
                        sens *= 0.1
                    } else if (mouse.modifiers & Qt.ControlModifier) {
                        sens *= 5.0
                    }
                    const newVal = root.roundToDecimals(root.clampValue(startVal + diff * sens))
                    root.valueEdited(newVal)
                }
            }

            onReleased: mouse => {
                if (!dragging) {
                    numberField.forceActiveFocus()
                    numberField.selectAll()
                }
                dragging = false
            }

            onCanceled: {
                dragging = false
            }
        }
    }
}
