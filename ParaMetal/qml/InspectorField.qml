pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    required property QtObject theme
    property var parameter: null
    property string label: parameter ? parameter.name : ""
    property string editor: "number"
    property var minimum: null
    property var maximum: null
    property real step: 0
    property int decimals: 3
    property var choices: []
    property bool compact: false
    signal valueEdited(var value)

    readonly property bool hasRange: (root.editor === "slider") || (minimum !== null && maximum !== null)

    function selectedChoiceText() {
        if (!root.parameter) return ""
        const value = String(root.parameter.value)
        const items = root.choices.length > 0 ? root.choices : root.parameter.options
        for (let index = 0; index < items.length; ++index) {
            const item = items[index]
            const itemValue = item.value === undefined ? item : item.value
            if (String(itemValue) === value)
                return String(item.text === undefined ? item : item.text)
        }
        return ""
    }

    implicitHeight: 28
    visible: parameter !== null && parameter !== undefined

    RowLayout {
        anchors.fill: parent
        spacing: root.theme.inspectorLabelSpacing

        Text {
            visible: root.label.length > 0
            Layout.preferredWidth: root.label.length === 0 ? 0 : (root.compact ? root.theme.inspectorCompactLabelWidth : root.theme.inspectorLabelWidth)
            text: root.label
            color: root.theme.text
            font: root.theme.regularFont
            elide: Text.ElideRight
        }

        UiNumberBox {
            id: numberBox
            theme: root.theme
            visible: root.editor === "number" || root.editor === "slider"
            Layout.fillWidth: true
            value: root.parameter ? Number(root.parameter.value) : 0
            minimum: root.minimum
            maximum: root.maximum
            hasRange: root.hasRange
            decimals: root.decimals
            step: root.step
            onValueEdited: val => root.valueEdited(val)
        }

        TextField {
            id: textField
            visible: root.editor === "text"
            Layout.fillWidth: true
            Layout.preferredHeight: 28
            text: root.parameter ? String(root.parameter.value) : ""
            color: root.theme.text
            selectionColor: root.theme.accent
            selectedTextColor: root.theme.onAccentText
            font: root.theme.regularFont
            padding: 7
            verticalAlignment: TextInput.AlignVCenter
            background: Rectangle {
                radius: 3
                color: root.theme.inputBackground
                border.width: 1
                border.color: textField.activeFocus ? root.theme.accent : root.theme.border
            }
            onEditingFinished: {
                if (root.parameter) root.valueEdited(text)
            }
        }

        UiCheckBox {
            theme: root.theme
            visible: root.editor === "bool"
            Layout.fillWidth: visible
            checked: root.parameter ? Boolean(root.parameter.value) : false
            text: ""
            onToggled: root.valueEdited(checked)
        }

        UiComboBox {
            id: combo
            theme: root.theme
            visible: root.editor === "combo"
            Layout.fillWidth: visible
            Layout.preferredHeight: 28
            model: root.choices.length > 0 ? root.choices : (root.parameter ? root.parameter.options : [])
            textRole: model && model.length > 0 && model[0].text !== undefined ? "text" : ""
            valueRole: model && model.length > 0 && model[0].value !== undefined ? "value" : ""
            currentIndex: {
                if (!root.parameter) return -1
                const value = String(root.parameter.value)
                for (let index = 0; index < model.length; ++index) {
                    const item = model[index]
                    if (String(item.value === undefined ? item : item.value) === value) return index
                }
                return -1
            }
            onActivated: {
                const item = model[currentIndex]
                root.valueEdited(item.value === undefined ? item : item.value)
            }
            contentItem: Text {
                leftPadding: 7
                text: root.selectedChoiceText()
                color: root.theme.text
                font: combo.font
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }
    }
}
