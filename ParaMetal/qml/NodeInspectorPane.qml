pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtQuick.Effects

Item {
    id: root
    required property QtObject theme
    required property QtObject graphModel
    required property QtObject runtimeNotifier
    property bool heatSolveActive: false
    property bool heatSolvePaused: false
    property string heatSolveErrorText: ""
    property string serialConnectionText: "Not used by an active Heat Solve"
    property string serialTemperatureText: "--"
    property string serialPollingRateText: "--"
    property var probeTemperatures: []
    property string exportStatusText: ""
    property bool exportStatusSuccess: true

    Connections {
        target: root.runtimeNotifier
        function onHeatSolveStatusChanged(active, paused) {
            root.heatSolveActive = active
            root.heatSolvePaused = paused
        }
        function onHeatSolveErrorChanged(error) {
            root.heatSolveErrorText = error
        }
        function onSerialStatusChanged(nodeId, connection, temperature, pollingRate) {
            if (graphModel.selectedNodeId === nodeId) {
                root.serialConnectionText = connection
                root.serialTemperatureText = temperature
                root.serialPollingRateText = pollingRate
            }
        }
        function onHeatProbeTemperaturesChanged(nodeId, temperatures) {
            if (graphModel.selectedNodeId === nodeId) {
                root.probeTemperatures = temperatures
            }
        }
        function onExportStatusChanged(nodeId, success, message) {
            if (graphModel.selectedNodeId === nodeId) {
                root.exportStatusText = message
                root.exportStatusSuccess = success
            }
        }
    }

    Connections {
        target: root.graphModel
        function onSelectedNodeChanged() {
            if (graphModel.selectedNodeId > 0) {
                root.probeTemperatures = root.runtimeNotifier.heatProbeTemperatures(graphModel.selectedNodeId)
                root.serialConnectionText = root.runtimeNotifier.serialConnectionText(graphModel.selectedNodeId)
                root.serialTemperatureText = root.runtimeNotifier.serialTemperatureText(graphModel.selectedNodeId)
                root.serialPollingRateText = root.runtimeNotifier.serialPollingRateText(graphModel.selectedNodeId)
                root.exportStatusText = root.runtimeNotifier.exportStatusMessage(graphModel.selectedNodeId)
                root.exportStatusSuccess = root.runtimeNotifier.exportStatusSuccess(graphModel.selectedNodeId)
            } else {
                root.probeTemperatures = []
                root.serialConnectionText = "Not used by an active Heat Solve"
                root.serialTemperatureText = "--"
                root.serialPollingRateText = "--"
                root.exportStatusText = ""
                root.exportStatusSuccess = true
            }
        }
    }

    Component.onCompleted: {
        root.heatSolveActive = root.runtimeNotifier.heatSolveActive()
        root.heatSolvePaused = root.runtimeNotifier.heatSolvePaused()
        root.heatSolveErrorText = root.runtimeNotifier.heatSolveError()
        if (graphModel.selectedNodeId > 0) {
            root.probeTemperatures = root.runtimeNotifier.heatProbeTemperatures(graphModel.selectedNodeId)
            root.serialConnectionText = root.runtimeNotifier.serialConnectionText(graphModel.selectedNodeId)
            root.serialTemperatureText = root.runtimeNotifier.serialTemperatureText(graphModel.selectedNodeId)
            root.serialPollingRateText = root.runtimeNotifier.serialPollingRateText(graphModel.selectedNodeId)
            root.exportStatusText = root.runtimeNotifier.exportStatusMessage(graphModel.selectedNodeId)
            root.exportStatusSuccess = root.runtimeNotifier.exportStatusSuccess(graphModel.selectedNodeId)
        }
    }

    function parameter(parameterId) {
        const parameters = graphModel.selectedNodeParameters
        for (let index = 0; index < parameters.length; ++index) {
            if (parameters[index].id === parameterId) return parameters[index]
        }
        return null
    }
    function setValue(parameterId, value) {
        root.exportStatusText = ""
        graphModel.setParameterValue(parameterId, value)
    }
    function panelForType(typeId) {
        switch (String(typeId)) {
        case "model": return modelPanel
        case "transform": return transformPanel
        case "group": return groupPanel
        case "remesh": return remeshPanel
        case "voronoi": return voronoiPanel
        case "heat_model": return heatModelPanel
        case "fluid":
        case "heat_fluid": return heatFluidPanel
        case "heat_solve": return heatSolvePanel
        case "boundary_conditions": return boundaryConditionsPanel
        case "serial_temperature": return serialTemperaturePanel
        case "heat_probe": return heatProbePanel
        case "points": return pointsPanel
        case "merge": return mergePanel
        case "write": return writePanel
        default: return genericPanel
        }
    }

    Flickable {
        id: inspectorScroll
        anchors.fill: parent
        clip: true
        contentWidth: width
        contentHeight: Math.max(height,
                                panelLoader.item ? panelLoader.item.implicitHeight + 40 : height)
        boundsBehavior: Flickable.StopAtBounds
        flickDeceleration: 4000
        maximumFlickVelocity: 900
        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
        }
        ScrollBar.horizontal: ScrollBar {
            policy: ScrollBar.AlwaysOff
        }

        Rectangle {
            x: 12
            width: Math.max(1, root.width - 24)
            height: inspectorScroll.contentHeight
            color: graphModel.selectedNodeId > 0 ? root.theme.cardBackground : root.theme.panelBackground
            border.width: graphModel.selectedNodeId > 0 ? 1 : 0
            border.color: root.theme.border
            radius: graphModel.selectedNodeId > 0 ? 10 : 0

            Text {
                visible: graphModel.selectedNodeId === 0
                x: 10; y: 2
                text: qsTr("This node has no editable actions yet")
                color: root.theme.text
                font: root.theme.regularFont
            }

            Loader {
                id: panelLoader
                visible: graphModel.selectedNodeId > 0
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 20
                sourceComponent: root.panelForType(graphModel.selectedNodeType)
            }
        }
    }

    Component {
        id: modelPanel
        Column {
            width: panelLoader.width
            spacing: root.theme.inspectorRowSpacing
            Text { text: qsTr("Model File:"); color: root.theme.text; font: root.theme.regularFont }
            RowLayout {
                width: parent.width
                TextField {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 29
                    readOnly: true
                    text: root.parameter(1) ? root.parameter(1).value : ""
                    color: root.theme.text
                    font: root.theme.regularFont
                    background: Rectangle { radius: 3; color: root.theme.inputBackground; border.width: 1; border.color: root.theme.border }
                }
                Button {
                    Layout.preferredWidth: 29
                    Layout.preferredHeight: 29
                    padding: 0
                    contentItem: Item {
                        Image {
                            anchors.centerIn: parent
                            width: 16
                            height: 16
                            source: "../textures/icons/File/open/32w/Artboard 1.png"
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    background: Rectangle {
                        radius: 4
                        color: parent.hovered ? root.theme.sourceButtonHover : root.theme.sourceButton
                        border.width: 1
                        border.color: root.theme.sourceButtonBorder
                    }
                    onClicked: modelDialog.open()
                }
            }
            FileDialog {
                id: modelDialog
                title: qsTr("Select Model File")
                nameFilters: [qsTr("3D Models (*.obj *.stl)"), qsTr("OBJ files (*.obj)"), qsTr("STL files (*.stl)"), qsTr("All files (*)")]
                onAccepted: {
                    let path = selectedFile.toString()
                    if (path.indexOf("file:///") === 0) path = path.substring(8)
                    root.setValue(1, decodeURIComponent(path))
                }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Import Units"); parameter: root.parameter(2); onValueEdited: value => root.setValue(2, value) }
        }
    }

    Component {
        id: transformPanel
        Column {
            width: panelLoader.width
            spacing: root.theme.inspectorGroupSpacing

            Column {
                width: parent.width
                spacing: root.theme.inspectorRowSpacing
                Text { text: qsTr("Translate"); color: root.theme.headingText; font: root.theme.regularFont }
                Column {
                    width: parent.width
                    spacing: root.theme.inspectorBoxSpacing
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(1); decimals: 4; onValueEdited: value => root.setValue(1, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(2); decimals: 4; onValueEdited: value => root.setValue(2, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(3); decimals: 4; onValueEdited: value => root.setValue(3, value) }
                }
            }

            Column {
                width: parent.width
                spacing: root.theme.inspectorRowSpacing
                Text { text: qsTr("Rotate"); color: root.theme.headingText; font: root.theme.regularFont }
                Column {
                    width: parent.width
                    spacing: root.theme.inspectorBoxSpacing
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(4); decimals: 3; onValueEdited: value => root.setValue(4, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(5); decimals: 3; onValueEdited: value => root.setValue(5, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(6); decimals: 3; onValueEdited: value => root.setValue(6, value) }
                }
            }

            Column {
                width: parent.width
                spacing: root.theme.inspectorRowSpacing
                Text { text: qsTr("Scale"); color: root.theme.headingText; font: root.theme.regularFont }
                Column {
                    width: parent.width
                    spacing: root.theme.inspectorBoxSpacing
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(7); decimals: 4; onValueEdited: value => root.setValue(7, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(8); decimals: 4; onValueEdited: value => root.setValue(8, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(9); decimals: 4; onValueEdited: value => root.setValue(9, value) }
                }
            }
        }
    }


    Component {
        id: groupPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Enable Assignment"); editor: "bool"; parameter: root.parameter(1); onValueEdited: value => root.setValue(1, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Source Type"); editor: "combo"; choices: [{text:"Vertex",value:0},{text:"Object",value:1},{text:"Material",value:2},{text:"Smooth",value:3}]; parameter: root.parameter(4); onValueEdited: value => root.setValue(4, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Source Name"); editor: "text"; parameter: root.parameter(2); onValueEdited: value => root.setValue(2, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Target Group Name"); editor: "text"; parameter: root.parameter(3); onValueEdited: value => root.setValue(3, value) }
            Text { width: parent.width; wrapMode: Text.Wrap; text: qsTr("Select a source type and source name from incoming mesh data, then assign triangles to a target group name."); color: root.theme.mutedText; font: root.theme.descriptionFont }
        }
    }

    Component {
        id: remeshPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            Column {
                width: parent.width; spacing: root.theme.inspectorBoxSpacing
                InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Iterations"); parameter: root.parameter(1); minimum: 1; maximum: 1000; decimals: 0; onValueEdited: value => root.setValue(1, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Min Angle"); parameter: root.parameter(2); minimum: 0; maximum: 60; decimals: 1; onValueEdited: value => root.setValue(2, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Max Edge Length"); parameter: root.parameter(3); minimum: .001; maximum: 1; decimals: 4; onValueEdited: value => root.setValue(3, value) }
                InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Step Size"); parameter: root.parameter(4); minimum: .01; maximum: 1; decimals: 2; onValueEdited: value => root.setValue(4, value) }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Remesh Overlay"); parameter: root.parameter(6); onValueEdited: value => root.setValue(6, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Face Normals"); parameter: root.parameter(7); onValueEdited: value => root.setValue(7, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Vertex Normals"); parameter: root.parameter(8); onValueEdited: value => root.setValue(8, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Normal Length"); parameter: root.parameter(9); minimum: .001; maximum: 1; decimals: 3; onValueEdited: value => root.setValue(9, value) }
        }
    }

    Component {
        id: voronoiPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            Column {
                width: parent.width; spacing: root.theme.inspectorBoxSpacing
                InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Cell Size"); parameter: root.parameter(1); minimum: .0001; maximum: .1; decimals: 6; onValueEdited: value => root.setValue(1, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Voxel Resolution"); parameter: root.parameter(2); minimum: 1; maximum: 1024; decimals: 0; onValueEdited: value => root.setValue(2, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Volume Grid Padding"); parameter: root.parameter(5); minimum: 0.0; maximum: .1; decimals: 4; onValueEdited: value => root.setValue(5, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Maximum Contact Gap"); parameter: root.parameter(6); minimum: .000001; maximum: .01; decimals: 6; onValueEdited: value => root.setValue(6, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Lloyd Iterations"); parameter: root.parameter(7); minimum: 0; maximum: 4; decimals: 0; onValueEdited: value => root.setValue(7, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "number"; label: qsTr("Lloyd Volume Threshold"); parameter: root.parameter(9); decimals: 8; onValueEdited: value => root.setValue(9, value) }
                InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Lloyd Damping"); parameter: root.parameter(10); minimum: .05; maximum: 1; decimals: 2; onValueEdited: value => root.setValue(10, value) }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Voronoi"); parameter: root.parameter(3); onValueEdited: value => root.setValue(3, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Points"); parameter: root.parameter(4); onValueEdited: value => root.setValue(4, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Contact Manifold"); parameter: root.parameter(8); onValueEdited: value => root.setValue(8, value) }
        }
    }

    Component {
        id: heatModelPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Material Preset"); parameter: root.parameter(7); onValueEdited: value => graphModel.setHeatMaterialPreset(value) }
            Column {
                width: parent.width; spacing: root.theme.inspectorBoxSpacing
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Initial Temperature (C)"); parameter: root.parameter(4); decimals: 2; onValueEdited: value => root.setValue(4, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Power Density (W/m³)"); parameter: root.parameter(10); onValueEdited: value => root.setValue(10, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Density (kg/m³)"); parameter: root.parameter(1); decimals: 1; onValueEdited: value => root.setValue(1, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Specific Heat (J/(kg·K))"); parameter: root.parameter(2); decimals: 1; onValueEdited: value => root.setValue(2, value) }
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Conductivity (W/(m·K))"); parameter: root.parameter(3); decimals: 2; onValueEdited: value => root.setValue(3, value) }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Fixed Temperature"); parameter: root.parameter(11); onValueEdited: value => root.setValue(11, value) }
            InspectorField { visible: root.parameter(11) && root.parameter(11).value; width: parent.width; theme: root.theme; label: qsTr("Temperature (C)"); parameter: root.parameter(12); decimals: 2; onValueEdited: value => root.setValue(12, value) }
        }
    }

    Component {
        id: heatFluidPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            Column {
                width: parent.width; spacing: root.theme.inspectorBoxSpacing
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Initial Temperature (C)"); parameter: root.parameter(4); decimals: 2; onValueEdited: value => root.setValue(4, value) }
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Density (kg/m³)"); parameter: root.parameter(1); decimals: 3; onValueEdited: value => root.setValue(1, value) }
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Specific Heat (J/(kg·K))"); parameter: root.parameter(2); decimals: 1; onValueEdited: value => root.setValue(2, value) }
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Conductivity (W/(m·K))"); parameter: root.parameter(3); decimals: 3; onValueEdited: value => root.setValue(3, value) }
                InspectorField { width: parent.width; theme: root.theme; label: qsTr("Kinematic Viscosity (m²/s)"); parameter: root.parameter(9); decimals: 8; onValueEdited: value => root.setValue(9, value) }
            }
        }
    }

    Component {
        id: heatSolvePanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            RowLayout {
                width: parent.width
                Text { text: qsTr("Status:"); color: root.theme.text; font: root.theme.regularFont }
                Text {
                    Layout.fillWidth: true
                    text: root.heatSolveActive
                        ? (root.heatSolvePaused ? qsTr("Paused") : qsTr("Running"))
                        : (root.parameter(1) && root.parameter(1).value
                            ? (root.parameter(2) && root.parameter(2).value ? qsTr("Pending Pause") : qsTr("Pending Start"))
                            : qsTr("Stopped"))
                    color: root.theme.text
                    font: root.theme.regularFont
                }
            }
            Column {
                width: parent.width; spacing: root.theme.inspectorBoxSpacing
                InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Simulation Duration (s)"); parameter: root.parameter(11); minimum: .1; maximum: 60; decimals: 1; onValueEdited: value => root.setValue(11, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Max PCG Iterations"); parameter: root.parameter(30); decimals: 0; onValueEdited: value => root.setValue(30, value) }
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("PCG Tolerance"); parameter: root.parameter(31); decimals: 6; onValueEdited: value => root.setValue(31, value) }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Heat Overlay"); parameter: root.parameter(5); onValueEdited: value => root.setValue(5, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Isocontours"); parameter: root.parameter(32); onValueEdited: value => root.setValue(32, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Isocontour Spacing (C)"); parameter: root.parameter(34); minimum: 0.1; maximum: 1000; decimals: 1; onValueEdited: value => root.setValue(34, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Flux Vectors"); parameter: root.parameter(7); onValueEdited: value => root.setValue(7, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Heat Palette"); parameter: root.parameter(9); onValueEdited: value => root.setValue(9, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Flux Vector Scale"); parameter: root.parameter(8); minimum: 0; maximum: 10; decimals: 2; onValueEdited: value => root.setValue(8, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Domain Box"); parameter: root.parameter(14); onValueEdited: value => root.setValue(14, value) }
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Show Fluid Volume"); parameter: root.parameter(15); onValueEdited: value => root.setValue(15, value) }
            InspectorField { visible: root.parameter(15) && root.parameter(15).value; width: parent.width; theme: root.theme; editor: "slider"; label: qsTr("Fluid Volume Opacity"); parameter: root.parameter(16); minimum: 0.01; maximum: 1.0; decimals: 2; onValueEdited: value => root.setValue(16, value) }

            Text {
                width: parent.width
                visible: root.heatSolveErrorText !== ""
                text: qsTr("Error: %1").arg(root.heatSolveErrorText)
                color: "#ff7a7a"
                font: root.theme.descriptionFont
                wrapMode: Text.Wrap
            }
        }
    }

    Component {
        id: boundaryConditionsPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing

            Repeater {
                model: [
                    { title: "-X", base: 100 },
                    { title: "+X", base: 110 },
                    { title: "-Y", base: 120 },
                    { title: "+Y", base: 130 },
                    { title: "-Z", base: 140 },
                    { title: "+Z", base: 150 }
                ]

                Column {
                    id: faceColumn
                    required property var modelData
                    readonly property int firstParameterId: modelData.base
                    readonly property string flow: root.parameter(firstParameterId) ? String(root.parameter(firstParameterId).value) : ""
                    readonly property string thermal: root.parameter(firstParameterId + 4) ? String(root.parameter(firstParameterId + 4).value) : ""

                    width: parent.width
                    spacing: root.theme.inspectorBoxSpacing

                    Text { text: faceColumn.modelData.title; color: root.theme.headingText; font: root.theme.regularFont; topPadding: 6 }
                    InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Flow"); parameter: root.parameter(faceColumn.firstParameterId); onValueEdited: value => root.setValue(faceColumn.firstParameterId, value) }
                    Column {
                        visible: faceColumn.flow === "Velocity Inlet"
                        width: parent.width
                        spacing: root.theme.inspectorRowSpacing

                        Text { text: qsTr("Velocity (m/s)"); color: root.theme.secondaryText; font: root.theme.regularFont }
                        Column {
                            width: parent.width
                            spacing: root.theme.inspectorBoxSpacing
                            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(faceColumn.firstParameterId + 1); decimals: 3; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 1, value) }
                            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(faceColumn.firstParameterId + 2); decimals: 3; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 2, value) }
                            InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(faceColumn.firstParameterId + 3); decimals: 3; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 3, value) }
                        }
                    }
                    InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Thermal"); parameter: root.parameter(faceColumn.firstParameterId + 4); onValueEdited: value => root.setValue(faceColumn.firstParameterId + 4, value) }
                    InspectorField { visible: faceColumn.thermal === "Dirichlet" || faceColumn.thermal === "Robin"; width: parent.width; theme: root.theme; label: faceColumn.thermal === "Robin" ? qsTr("Ambient Temp (C)") : qsTr("Temperature (C)"); parameter: root.parameter(faceColumn.firstParameterId + 5); decimals: 2; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 5, value) }
                    InspectorField { visible: faceColumn.thermal === "Neumann"; width: parent.width; theme: root.theme; label: qsTr("Heat Flux (W/m²)"); parameter: root.parameter(faceColumn.firstParameterId + 6); decimals: 2; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 6, value) }
                    InspectorField { visible: faceColumn.thermal === "Robin"; width: parent.width; theme: root.theme; label: qsTr("HTC (W/(m²·K))"); parameter: root.parameter(faceColumn.firstParameterId + 7); decimals: 2; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 7, value) }
                    InspectorField { visible: faceColumn.flow === "Velocity Inlet" || faceColumn.flow === "Pressure Outlet"; width: parent.width; theme: root.theme; label: qsTr("Inflow Temperature (C)"); parameter: root.parameter(faceColumn.firstParameterId + 8); decimals: 2; onValueEdited: value => root.setValue(faceColumn.firstParameterId + 8, value) }
                }
            }
        }
    }

    Component {
        id: serialTemperaturePanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            InspectorField { width: parent.width; theme: root.theme; editor: "bool"; label: qsTr("Enabled"); parameter: root.parameter(1); onValueEdited: value => root.setValue(1, value) }
            RowLayout {
                width: parent.width
                InspectorField { Layout.fillWidth: true; theme: root.theme; editor: "combo"; label: qsTr("Port"); parameter: root.parameter(2); choices: graphModel.serialPorts; onValueEdited: value => root.setValue(2, value) }
                Button { text: qsTr("Refresh"); onClicked: graphModel.refreshSerialPorts() }
            }
            InspectorField { width: parent.width; theme: root.theme; editor: "combo"; label: qsTr("Baud Rate"); parameter: root.parameter(3); choices: [9600,19200,38400,57600,115200,230400]; onValueEdited: value => root.setValue(3, value) }
            RowLayout {
                width: parent.width
                Text { text: qsTr("Status:"); color: root.theme.text; font: root.theme.regularFont }
                Text { Layout.fillWidth: true; text: root.serialConnectionText; color: root.theme.text; font: root.theme.regularFont; wrapMode: Text.Wrap }
            }
            RowLayout {
                width: parent.width
                Text { text: qsTr("Latest Temperature:"); color: root.theme.text; font: root.theme.regularFont }
                Text { Layout.fillWidth: true; text: root.serialTemperatureText; color: root.theme.text; font: root.theme.regularFont }
            }
            RowLayout {
                width: parent.width
                Text { text: qsTr("Polling Rate:"); color: root.theme.text; font: root.theme.regularFont }
                Text { Layout.fillWidth: true; text: root.serialPollingRateText; color: root.theme.text; font: root.theme.regularFont }
            }
        }
    }

    Component {
        id: pointsPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            InspectorField { width: parent.width; theme: root.theme; label: qsTr("Point Count"); parameter: root.parameter(1); decimals: 0; onValueEdited: value => root.setValue(1, value) }
            Column {
                width: parent.width
                spacing: root.theme.inspectorRowSpacing
                Text { text: qsTr("Dimensions"); color: root.theme.headingText; font: root.theme.regularFont }
                Column {
                    width: parent.width
                    spacing: root.theme.inspectorBoxSpacing
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "X"; parameter: root.parameter(2); decimals: 6; onValueEdited: value => root.setValue(2, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Y"; parameter: root.parameter(3); decimals: 6; onValueEdited: value => root.setValue(3, value) }
                    InspectorField { width: parent.width; theme: root.theme; compact: true; label: "Z"; parameter: root.parameter(4); decimals: 6; onValueEdited: value => root.setValue(4, value) }
                }
            }
        }
    }

    Component {
        id: mergePanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            InspectorField {
                width: parent.width
                theme: root.theme
                label: qsTr("Merge Distance")
                parameter: root.parameter(1)
                decimals: 6
                minimum: 0.0
                maximum: 10.0
                onValueEdited: value => root.setValue(1, value)
            }
        }
    }

    Component {
        id: writePanel
        Column {
            id: writeCol
            width: panelLoader.width
            spacing: root.theme.inspectorRowSpacing
            property string lastCommittedPath: graphModel.selectedWriteFile()

            function fileDialogUrl(path) {
                const normalized = String(path).replace(/\\/g, "/")
                if (/^[A-Za-z]:\//.test(normalized)) return "file:///" + normalized
                if (normalized.startsWith("/")) return "file://" + normalized
                return "file:///"
            }

            function commitPath(path) {
                if (graphModel.selectedNodeType !== "write") return;
                if (path !== lastCommittedPath) {
                    lastCommittedPath = path
                    root.setValue(1, path)
                    if (graphModel.selectedNodeId > 0) {
                        root.runtimeNotifier.clearExportStatus(graphModel.selectedNodeId)
                    }
                }
            }

            Connections {
                target: root.graphModel
                function onSelectedNodeChanged() {
                    const currentVal = graphModel.selectedWriteFile()
                    writeCol.lastCommittedPath = currentVal
                    if (!writePathField.activeFocus) writePathField.text = currentVal
                }
            }

            Text {
                text: qsTr("File:")
                color: root.theme.text
                font: root.theme.regularFont
            }
            RowLayout {
                width: parent.width
                spacing: 7
                TextField {
                    id: writePathField
                    Layout.fillWidth: true
                    Layout.preferredHeight: 29
                    text: graphModel.selectedWriteFile()
                    color: root.theme.text
                    font: root.theme.regularFont
                    background: Rectangle {
                        radius: 3
                        color: root.theme.inputBackground
                        border.width: 1
                        border.color: root.theme.border
                    }
                    onEditingFinished: commitPath(text)
                }
                Button {
                    Layout.preferredWidth: 29
                    Layout.preferredHeight: 29
                    padding: 0
                    contentItem: Item {
                        Image {
                            anchors.centerIn: parent
                            width: 16
                            height: 16
                            source: "../textures/icons/File/open/32w/Artboard 1.png"
                            fillMode: Image.PreserveAspectFit
                        }
                    }
                    background: Rectangle {
                        radius: 4
                        color: parent.hovered ? root.theme.sourceButtonHover : root.theme.sourceButton
                        border.width: 1
                        border.color: root.theme.sourceButtonBorder
                    }
                    onClicked: exportFileDialog.open()
                }
            }
            FileDialog {
                id: exportFileDialog
                title: qsTr("Select Export File")
                fileMode: FileDialog.SaveFile
                defaultSuffix: graphModel.exportProductFormat.toLowerCase()
                currentFile: writeCol.fileDialogUrl(writePathField.text)
                onAccepted: {
                    let path = selectedFile.toString()
                    if (path.indexOf("file:///") === 0) {
                        const drivePath = /^[A-Za-z]:/.test(path.substring(8))
                        path = path.substring(drivePath ? 8 : 7)
                    }
                    const decoded = decodeURIComponent(path)
                    writePathField.text = decoded
                    commitPath(decoded)
                }
            }

            RowLayout {
                width: parent.width
                spacing: root.theme.inspectorLabelSpacing
                Text {
                    text: qsTr("Product")
                    color: root.theme.text
                    font: root.theme.regularFont
                }
                UiComboBox {
                    id: exportProductCombo
                    theme: root.theme
                    Layout.fillWidth: true
                    model: graphModel.exportProductLabels
                    currentIndex: graphModel.selectedExportProductIndex
                    placeholderText: qsTr("None")
                    onActivated: graphModel.selectExportProduct(currentIndex)
                }
            }

            Text {
                width: parent.width
                visible: graphModel.exportProductsError !== ""
                text: graphModel.exportProductsError
                color: "#ff7a7a"
                font: root.theme.regularFont
                wrapMode: Text.Wrap
            }

            Text {
                width: parent.width
                visible: graphModel.exportProductsError === "" && graphModel.exportProductLabels.length === 0
                text: qsTr("No exportable products found upstream")
                color: root.theme.text
                font: root.theme.regularFont
            }

            Text {
                width: parent.width
                visible: graphModel.exportProductStatus !== ""
                text: graphModel.exportProductStatus
                color: "#ff7a7a"
                font: root.theme.regularFont
                wrapMode: Text.Wrap
            }

            RowLayout {
                visible: graphModel.selectedExportProductIndex >= 0
                width: parent.width
                Text { text: qsTr("Format"); color: root.theme.text; font: root.theme.regularFont }
                UiComboBox {
                    theme: root.theme
                    Layout.preferredWidth: 92
                    model: graphModel.exportProductFormats
                    currentIndex: Math.max(0, model.indexOf(graphModel.exportProductFormat))
                    onActivated: graphModel.setExportProductFormat(currentText)
                }
                Item { Layout.fillWidth: true }
            }

            Item { width: parent.width; height: 6 }

            Button {
                width: 88
                height: 28
                enabled: graphModel.exportProductAvailable && writePathField.text.trim().length > 0
                text: qsTr("Export")
                font: root.theme.regularFont
                background: Rectangle {
                    radius: 4
                    color: parent.down ? root.theme.sourceButtonHover : (parent.hovered ? root.theme.sourceButtonHover : root.theme.sourceButton)
                    border.width: 1
                    border.color: root.theme.sourceButtonBorder
                }
                contentItem: Text {
                    text: parent.text
                    color: root.theme.text
                    opacity: parent.enabled ? 1.0 : 0.45
                    font: parent.font
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                onClicked: {
                    writePathField.focus = false
                    commitPath(writePathField.text)
                    graphModel.exportNode(graphModel.selectedNodeId)
                }
            }

            Text {
                width: parent.width
                visible: root.exportStatusText !== ""
                text: root.exportStatusText
                color: root.exportStatusSuccess ? root.theme.accent : "#ff7a7a"
                font: root.theme.regularFont
                wrapMode: Text.Wrap
            }
        }
    }

    Component {
        id: heatProbePanel
        Column {
            id: probeCol
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            property bool isPicking: graphModel ? graphModel.probePickingActive : false
            property int selectedProbeIndex: 0
            property string tempUnit: "C"

            function getProbesList() {
                const param = root.parameter(3)
                if (!param || !param.value) {
                    return [{ id: 1, name: "Probe_1", x: 0.0, y: 0.0, z: 0.0, enabled: true }]
                }
                try {
                    const parsed = typeof param.value === "string" ? JSON.parse(param.value) : param.value
                    if (Array.isArray(parsed) && parsed.length > 0) return parsed
                } catch (e) {
                }
                return [{ id: 1, name: "Probe_1", x: 0.0, y: 0.0, z: 0.0, enabled: true }]
            }

            function saveProbes(list) {
                root.setValue(3, JSON.stringify(list))
            }

            readonly property var probes: getProbesList()

            // Blender-style UIList container: List on left, [ + ], [ - ] and Marker on right
            RowLayout {
                width: parent.width
                spacing: 6

                // List Box
                Rectangle {
                    id: probeListBox
                    Layout.fillWidth: true
                    implicitHeight: Math.max(96, Math.min(180, probeRepeater.count * 28 + 8))
                    radius: 4
                    color: root.theme.inputBackground
                    border.color: probeCol.isPicking ? root.theme.accent : root.theme.subtleBorder
                    border.width: probeCol.isPicking ? 2 : 1
                    clip: true

                    Flickable {
                        anchors.fill: parent
                        anchors.margins: 4
                        contentHeight: probeRowsCol.implicitHeight
                        clip: true
                        boundsBehavior: Flickable.StopAtBounds

                        Column {
                            id: probeRowsCol
                            width: parent.width
                            spacing: 2

                            Repeater {
                                id: probeRepeater
                                model: probeCol.probes

                                Rectangle {
                                    id: rowRect
                                    required property var modelData
                                    required property int index

                                    width: probeRowsCol.width
                                    height: 26
                                    radius: 3
                                    color: (index === probeCol.selectedProbeIndex)
                                        ? Qt.rgba(root.theme.accent.r, root.theme.accent.g, root.theme.accent.b, 0.28)
                                        : (rowMouse.containsMouse ? Qt.rgba(1, 1, 1, 0.05) : "transparent")

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 4
                                        anchors.rightMargin: 8
                                        spacing: 6

                                        // Inline editable probe name
                                        TextField {
                                            id: inlineNameEdit
                                            Layout.fillWidth: true
                                            Layout.preferredHeight: 22
                                            text: rowRect.modelData.name || ("Probe_" + (rowRect.index + 1))
                                            color: (rowRect.index === probeCol.selectedProbeIndex) ? "#ffffff" : root.theme.text
                                            font: root.theme.regularFont
                                            padding: 1
                                            leftPadding: 4
                                            background: Rectangle {
                                                color: inlineNameEdit.activeFocus ? root.theme.inputBackground : "transparent"
                                                border.width: inlineNameEdit.activeFocus ? 1 : 0
                                                border.color: root.theme.border
                                                radius: 2
                                            }
                                            onPressed: {
                                                probeCol.selectedProbeIndex = rowRect.index
                                            }
                                            onEditingFinished: {
                                                if (text.trim().length > 0 && text.trim() !== (rowRect.modelData.name || "")) {
                                                    let list = probeCol.getProbesList()
                                                    if (rowRect.index >= 0 && rowRect.index < list.length) {
                                                        list[rowRect.index].name = text.trim()
                                                        probeCol.saveProbes(list)
                                                    }
                                                }
                                            }
                                        }

                                        // Temperature readout: exact same font and color as probe name
                                        Text {
                                            readonly property real tempC: (root.probeTemperatures && rowRect.index < root.probeTemperatures.length)
                                                ? root.probeTemperatures[rowRect.index]
                                                : 20.0
                                            readonly property real displayVal: (probeCol.tempUnit === "F") ? (tempC * 1.8 + 32.0) : tempC
                                            text: displayVal.toFixed(2)
                                            color: (rowRect.index === probeCol.selectedProbeIndex) ? "#ffffff" : root.theme.text
                                            font: root.theme.regularFont
                                        }

                                        // Clickable unit toggle right to the right of the temperature value
                                        Text {
                                            text: probeCol.tempUnit
                                            color: (rowRect.index === probeCol.selectedProbeIndex) ? "#ffffff" : root.theme.secondaryText
                                            font: root.theme.regularFont

                                            MouseArea {
                                                anchors.fill: parent
                                                cursorShape: Qt.PointingHandCursor
                                                ToolTip.visible: containsMouse
                                                ToolTip.text: qsTr("Toggle unit (C / F)")
                                                onClicked: {
                                                    probeCol.tempUnit = (probeCol.tempUnit === "C") ? "F" : "C"
                                                }
                                            }
                                        }
                                    }

                                    MouseArea {
                                        id: rowMouse
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        z: -1
                                        onClicked: {
                                            probeCol.selectedProbeIndex = rowRect.index
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Vertical action column on the right: [ + ], [ - ] at top, Marker icon at bottom
                ColumnLayout {
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: probeListBox.implicitHeight
                    Layout.alignment: Qt.AlignTop
                    spacing: 4

                    // Add probe (+)
                    Button {
                        Layout.preferredWidth: 28
                        Layout.preferredHeight: 28
                        text: "+"
                        font.bold: true
                        font.pixelSize: 15
                        padding: 0
                        ToolTip.visible: hovered
                        ToolTip.text: qsTr("Add probe")

                        contentItem: Text {
                            text: parent.text
                            color: root.theme.text
                            font: parent.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            radius: 4
                            color: parent.hovered ? root.theme.sourceButtonHover : root.theme.sourceButton
                            border.width: 1
                            border.color: root.theme.sourceButtonBorder
                        }
                        onClicked: {
                            let list = probeCol.getProbesList()
                            let newId = (list.length > 0) ? (list[list.length - 1].id + 1) : 1
                            list.push({
                                id: newId,
                                name: "Probe_" + newId,
                                x: 0.0,
                                y: 0.0,
                                z: 0.0,
                                enabled: true
                            })
                            probeCol.saveProbes(list)
                            probeCol.selectedProbeIndex = list.length - 1
                        }
                    }

                    // Remove probe (-)
                    Button {
                        Layout.preferredWidth: 28
                        Layout.preferredHeight: 28
                        text: "−"
                        font.bold: true
                        font.pixelSize: 15
                        padding: 0
                        enabled: probeCol.probes.length > 1
                        opacity: enabled ? 1.0 : 0.4
                        ToolTip.visible: hovered
                        ToolTip.text: qsTr("Remove selected probe")

                        contentItem: Text {
                            text: parent.text
                            color: root.theme.text
                            font: parent.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            radius: 4
                            color: parent.hovered ? root.theme.sourceButtonHover : root.theme.sourceButton
                            border.width: 1
                            border.color: root.theme.sourceButtonBorder
                        }
                        onClicked: {
                            let list = probeCol.getProbesList()
                            if (list.length > 1) {
                                list.splice(probeCol.selectedProbeIndex, 1)
                                probeCol.selectedProbeIndex = Math.max(0, Math.min(probeCol.selectedProbeIndex, list.length - 1))
                                probeCol.saveProbes(list)
                            }
                        }
                    }

                    // Flexible spacer pushing marker icon to the bottom right of the box
                    Item {
                        Layout.fillHeight: true
                    }

                    // Frameless Marker Icon at the bottom right of the probe box
                    Item {
                        Layout.preferredWidth: 28
                        Layout.preferredHeight: 28
                        Layout.alignment: Qt.AlignHCenter

                        Image {
                            id: markerIconImg
                            anchors.centerIn: parent
                            width: 20
                            height: 20
                            source: "../textures/icons/Marker/marker_a/128w/Artboard 1.png"
                            fillMode: Image.PreserveAspectFit
                            smooth: true
                            mipmap: true
                            opacity: probeCol.isPicking ? 1.0 : (markerMouse.containsMouse ? 0.95 : 0.6)
                            layer.enabled: true
                            layer.effect: MultiEffect {
                                colorization: 1.0
                                colorizationColor: probeCol.isPicking ? root.theme.accent : root.theme.toolIconColor
                            }
                        }

                        Rectangle {
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 1
                            anchors.horizontalCenter: parent.horizontalCenter
                            width: 4
                            height: 4
                            radius: 2
                            color: root.theme.accent
                            visible: probeCol.isPicking
                        }

                        MouseArea {
                            id: markerMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            ToolTip.visible: containsMouse
                            ToolTip.text: probeCol.isPicking
                                ? qsTr("Cancel marker picking")
                                : qsTr("Pick surface coordinate on 3D model")
                            onClicked: {
                                if (probeCol.isPicking) {
                                    graphModel.cancelProbePicking()
                                } else {
                                    graphModel.startProbePicking(graphModel.selectedNodeId, probeCol.selectedProbeIndex)
                                }
                            }
                        }
                    }
                }
            }

            // Port parameter (aligned with probe list box)
            RowLayout {
                width: parent.width
                spacing: root.theme.inspectorLabelSpacing

                InspectorField {
                    Layout.fillWidth: true
                    theme: root.theme
                    label: qsTr("Port")
                    parameter: root.parameter(2)
                    decimals: 0
                    onValueEdited: value => root.setValue(2, value)
                }

                Item {
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 1
                }
            }

            // Display flags: Show Temperature and Show Name
            RowLayout {
                width: parent.width
                spacing: root.theme.inspectorLabelSpacing

                InspectorField {
                    Layout.fillWidth: true
                    theme: root.theme
                    editor: "bool"
                    label: qsTr("Show Temperature")
                    parameter: root.parameter(4)
                    onValueEdited: value => root.setValue(4, value)
                }

                Item {
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 1
                }
            }

            RowLayout {
                width: parent.width
                spacing: root.theme.inspectorLabelSpacing

                InspectorField {
                    Layout.fillWidth: true
                    theme: root.theme
                    editor: "bool"
                    label: qsTr("Show Name")
                    parameter: root.parameter(5)
                    onValueEdited: value => root.setValue(5, value)
                }

                Item {
                    Layout.preferredWidth: 28
                    Layout.preferredHeight: 1
                }
            }
        }
    }

    Component {
        id: genericPanel
        Column {
            width: panelLoader.width; spacing: root.theme.inspectorRowSpacing
            Text { visible: graphModel.selectedNodeParameters.length === 0; text: qsTr("This node has no editable actions yet"); color: root.theme.text; font: root.theme.regularFont }
            Column {
                visible: graphModel.selectedNodeParameters.length > 0
                width: parent.width
                spacing: root.theme.inspectorBoxSpacing
                Repeater {
                    model: graphModel.selectedNodeParameters
                    InspectorField {
                        required property var modelData
                        width: parent.width
                        theme: root.theme
                        parameter: modelData
                        editor: modelData.type === 2 ? "bool" : (modelData.type === 4 ? "combo" : (modelData.type === 3 ? "text" : "number"))
                        onValueEdited: value => root.setValue(modelData.id, value)
                    }
                }
            }
        }
    }
}
