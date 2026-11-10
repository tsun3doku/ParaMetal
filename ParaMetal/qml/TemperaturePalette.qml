pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

FloatingPanel {
    id: root
    required property QtObject heatPalette

    title: qsTr("Temperature")
    panelModel: heatPalette

    function paletteSource() {
        if (heatPalette.paletteId === 0) return "../textures/palettes/inferno2.png"
        if (heatPalette.paletteId === 1) return "../textures/palettes/parula.png"
        if (heatPalette.paletteId === 2) return "../textures/palettes/viridis.png"
        return "../textures/palettes/inferno.png"
    }
    function toDisplay(c) {
        if (heatPalette.units === 1) return c + 273.15
        if (heatPalette.units === 2) return c * 9 / 5 + 32
        return c
    }
    function toCelsius(value) {
        if (heatPalette.units === 1) return value - 273.15
        if (heatPalette.units === 2) return (value - 32) * 5 / 9
        return value
    }

    RowLayout {
        Layout.fillWidth: true; spacing: 6
        Text { text: qsTr("Palette"); color: theme.text; font: theme.regularFont; Layout.preferredWidth: 48 }
        UiComboBox {
            theme: root.theme; Layout.fillWidth: true; Layout.preferredHeight: 28
            model: [
                { text: qsTr("Inferno 2"), value: 0 },
                { text: qsTr("Parula"), value: 1 },
                { text: qsTr("Viridis"), value: 2 },
                { text: qsTr("Inferno"), value: 3 }
            ]
            textRole: "text"
            valueRole: "value"
            currentIndex: {
                for (let i = 0; i < model.length; ++i) {
                    if (model[i].value === heatPalette.paletteId) return i
                }
                return 0
            }
            onActivated: heatPalette.setPaletteId(model[currentIndex].value)
        }
    }

    RowLayout {
        Layout.fillWidth: true; spacing: 6
        Text { text: qsTr("Units"); color: theme.text; font: theme.regularFont; Layout.preferredWidth: 48 }
        UiComboBox {
            theme: root.theme; Layout.fillWidth: true; Layout.preferredHeight: 28
            model: [qsTr("°C"), qsTr("K"), qsTr("°F")]
            currentIndex: heatPalette.units
            onActivated: heatPalette.setUnits(currentIndex)
        }
    }

    RowLayout {
        Layout.fillWidth: true; Layout.fillHeight: true; spacing: 6

        Image {
            Layout.preferredWidth: 48; Layout.fillHeight: true
            source: root.paletteSource(); fillMode: Image.Stretch; smooth: false
        }

        ColumnLayout {
            Layout.fillWidth: true; Layout.fillHeight: true; spacing: 0

            TextField {
                id: maxField; Layout.fillWidth: true; Layout.preferredHeight: 28
                text: root.toDisplay(root.heatPalette.maximumC).toFixed(1)
                color: root.theme.text; font: root.theme.regularFont; padding: 6
                verticalAlignment: TextInput.AlignVCenter; validator: DoubleValidator {}
                background: Rectangle { radius: 3; color: root.theme.cardBackground; border.width: 1; border.color: maxField.activeFocus ? root.theme.accent : root.theme.border }
                onEditingFinished: { const v = Number(text); if (isFinite(v)) root.heatPalette.setRange(root.heatPalette.minimumC, root.toCelsius(v)) }
            }

            Item { Layout.fillWidth: true; Layout.fillHeight: true }

            TextField {
                id: minField; Layout.fillWidth: true; Layout.preferredHeight: 28
                text: root.toDisplay(root.heatPalette.minimumC).toFixed(1)
                color: root.theme.text; font: root.theme.regularFont; padding: 6
                verticalAlignment: TextInput.AlignVCenter; validator: DoubleValidator {}
                background: Rectangle { radius: 3; color: root.theme.cardBackground; border.width: 1; border.color: minField.activeFocus ? root.theme.accent : root.theme.border }
                onEditingFinished: { const v = Number(text); if (isFinite(v)) root.heatPalette.setRange(root.toCelsius(v), root.heatPalette.maximumC) }
            }
        }
    }
}
