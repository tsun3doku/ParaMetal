pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

FloatingPanel {
    id: root
    required property QtObject velocityPalette

    title: qsTr("Velocity")
    panelModel: velocityPalette

    function paletteSource() {
        if (velocityPalette.paletteId === 0) return "../textures/palettes/inferno2.png"
        if (velocityPalette.paletteId === 1) return "../textures/palettes/parula.png"
        if (velocityPalette.paletteId === 2) return "../textures/palettes/viridis.png"
        return "../textures/palettes/inferno.png"
    }
    function decimals() { return velocityPalette.units === 0 ? 3 : 1 }
    function toDisplay(ms) {
        if (velocityPalette.units === 1) return ms * 100
        if (velocityPalette.units === 2) return ms * 1000
        return ms
    }
    function toMeters(value) {
        if (velocityPalette.units === 1) return value / 100
        if (velocityPalette.units === 2) return value / 1000
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
                    if (model[i].value === velocityPalette.paletteId) return i
                }
                return 0
            }
            onActivated: velocityPalette.setPaletteId(model[currentIndex].value)
        }
    }

    RowLayout {
        Layout.fillWidth: true; spacing: 6
        Text { text: qsTr("Units"); color: theme.text; font: theme.regularFont; Layout.preferredWidth: 48 }
        UiComboBox {
            theme: root.theme; Layout.fillWidth: true; Layout.preferredHeight: 28
            model: [qsTr("m/s"), qsTr("cm/s"), qsTr("mm/s")]
            currentIndex: velocityPalette.units
            onActivated: velocityPalette.setUnits(currentIndex)
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
                text: root.toDisplay(root.velocityPalette.maximum).toFixed(root.decimals())
                color: root.theme.text; font: root.theme.regularFont; padding: 6
                verticalAlignment: TextInput.AlignVCenter; validator: DoubleValidator {}
                background: Rectangle { radius: 3; color: root.theme.cardBackground; border.width: 1; border.color: maxField.activeFocus ? root.theme.accent : root.theme.border }
                onEditingFinished: { const v = Number(text); if (isFinite(v)) root.velocityPalette.setRange(root.velocityPalette.minimum, root.toMeters(v)) }
            }

            Item { Layout.fillWidth: true; Layout.fillHeight: true }

            TextField {
                id: minField; Layout.fillWidth: true; Layout.preferredHeight: 28
                text: root.toDisplay(root.velocityPalette.minimum).toFixed(root.decimals())
                color: root.theme.text; font: root.theme.regularFont; padding: 6
                verticalAlignment: TextInput.AlignVCenter; validator: DoubleValidator {}
                background: Rectangle { radius: 3; color: root.theme.cardBackground; border.width: 1; border.color: minField.activeFocus ? root.theme.accent : root.theme.border }
                onEditingFinished: { const v = Number(text); if (isFinite(v)) root.velocityPalette.setRange(root.toMeters(v), root.velocityPalette.maximum) }
            }
        }
    }
}
