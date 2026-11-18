pragma ComponentBehavior: Bound

import QtQuick

Column {
    id: root
    required property QtObject theme
    required property QtObject graphModel
    required property int base

    readonly property string flowValue: {
        const parameter = root.parameter(root.base)
        return parameter ? String(parameter.value) : ""
    }
    readonly property string thermalValue: {
        const parameter = root.parameter(root.base + 4)
        return parameter ? String(parameter.value) : ""
    }

    width: parent ? parent.width : implicitWidth
    spacing: theme.inspectorRowSpacing

    function parameter(parameterId) {
        const parameters = graphModel.selectedNodeParameters
        for (let index = 0; index < parameters.length; ++index) {
            if (parameters[index].id === parameterId) return parameters[index]
        }
        return null
    }
    function setValue(parameterId, value) {
        graphModel.setParameterValue(parameterId, value)
    }

    InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Flow"); parameter: root.parameter(root.base); onValueEdited: value => root.setValue(root.base, value) }
    Column {
        visible: root.flowValue === "Velocity Inlet"
        width: parent.width
        spacing: root.theme.inspectorRowSpacing

        Text { text: qsTr("Velocity (m/s)"); color: root.theme.secondaryText; font: root.theme.regularFont }
        Column {
            width: parent.width
            spacing: root.theme.inspectorBoxSpacing
            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(root.base + 1); decimals: 3; onValueEdited: value => root.setValue(root.base + 1, value) }
            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(root.base + 2); decimals: 3; onValueEdited: value => root.setValue(root.base + 2, value) }
            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(root.base + 3); decimals: 3; onValueEdited: value => root.setValue(root.base + 3, value) }
        }
    }
    InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Thermal"); parameter: root.parameter(root.base + 4); onValueEdited: value => root.setValue(root.base + 4, value) }
    InspectorField { visible: root.thermalValue === "Dirichlet" || root.thermalValue === "Robin"; width: parent.width; theme: root.theme; label: root.thermalValue === "Robin" ? qsTr("Ambient Temp (C)") : qsTr("Temperature (C)"); parameter: root.parameter(root.base + 5); decimals: 2; onValueEdited: value => root.setValue(root.base + 5, value) }
    InspectorField { visible: root.thermalValue === "Neumann"; width: parent.width; theme: root.theme; label: qsTr("Heat Flux (W/m²)"); parameter: root.parameter(root.base + 6); decimals: 2; onValueEdited: value => root.setValue(root.base + 6, value) }
    InspectorField { visible: root.thermalValue === "Robin"; width: parent.width; theme: root.theme; label: qsTr("HTC (W/(m²·K))"); parameter: root.parameter(root.base + 7); decimals: 2; onValueEdited: value => root.setValue(root.base + 7, value) }
    InspectorField { visible: root.flowValue === "Velocity Inlet" || root.flowValue === "Pressure Outlet"; width: parent.width; theme: root.theme; label: qsTr("Inflow Temperature (C)"); parameter: root.parameter(root.base + 8); decimals: 2; onValueEdited: value => root.setValue(root.base + 8, value) }
}
