#include "NodeHeatSolverPanel.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"

#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "NodePanelUtils.hpp"

#include <iostream>
#include "NodeGraphSliderRow.hpp"
#include "NodeGraphWidgetStyle.hpp"
#include "runtime/RuntimeInterfaces.hpp"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>

#include <string>

NodeHeatSolverPanel::NodeHeatSolverPanel(QWidget* parent)
    : NodePanelBase(parent) {
    QVBoxLayout* layout = static_cast<QVBoxLayout*>(this->layout());

    QHBoxLayout* statusRow = new QHBoxLayout();
    statusRow->addWidget(new QLabel("Status:", this));
    heatStatusValueLabel = new QLabel("Unknown", this);
    statusRow->addWidget(heatStatusValueLabel, 1);
    layout->addLayout(statusRow);

    layout->addWidget(new QLabel("Solver Settings:", this));

    heatContactThermalConductanceRow = new NodeGraphSliderRow("Contact Thermal Conductance", this);
    heatContactThermalConductanceRow->setRange(0.0, 100000.0);
    heatContactThermalConductanceRow->setDecimals(0);
    heatContactThermalConductanceRow->setValue(16000.0);
    layout->addWidget(heatContactThermalConductanceRow);

    heatSimulationDurationRow = new NodeGraphSliderRow("Simulation Duration (s)", this);
    heatSimulationDurationRow->setRange(0.1, 60.0);
    heatSimulationDurationRow->setDecimals(1);
    heatSimulationDurationRow->setValue(5.0);
    layout->addWidget(heatSimulationDurationRow);

    heatOverlayCheckBox = new QCheckBox("Heat Overlay", this);
    layout->addWidget(heatOverlayCheckBox);

    fluxVectorsCheckBox = new QCheckBox("Flux Vectors", this);
    layout->addWidget(fluxVectorsCheckBox);

    heatPaletteCheckBox = new QCheckBox("Heat Palette", this);
    layout->addWidget(heatPaletteCheckBox);

    fluxVectorScaleRow = new NodeGraphSliderRow("Flux Vector Scale", this);
    fluxVectorScaleRow->setRange(0.0, 10.0);
    fluxVectorScaleRow->setDecimals(2);
    fluxVectorScaleRow->setValue(1.0);
    layout->addWidget(fluxVectorScaleRow);

    layout->addWidget(new QLabel("Contact parameters are authored on the Contact node.", this));
    layout->addStretch();

    heatStatusTimer = new QTimer(this);
    heatStatusTimer->setInterval(nodegraphwidgets::heatStatusTimerIntervalMs);

    heatContactThermalConductanceRow->setValueChangedCallback([this](double) { onSolverSettingsEdited(); });
    heatSimulationDurationRow->setValueChangedCallback([this](double) { onSolverSettingsEdited(); });
    connect(heatOverlayCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (isSyncing()) {
            return;
        }

        HeatSolveNodeParams params{};
        if (!tryLoadNodeParams(params)) {
            std::cerr << "[HeatSolverPanel] Failed to load node params for overlay toggle" << std::endl;
            setStatus("Cannot update heat overlay settings for this node.");
            return;
        }

        params.preview.showHeatOverlay = checked;
        if (!writeNodeParams(params)) {
            std::cerr << "[HeatSolverPanel] Failed to write node params for overlay toggle" << std::endl;
            setStatus("Failed to update heat overlay settings.");
            return;
        }

        setStatus("Heat settings applied.");
    });
    connect(fluxVectorsCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (isSyncing()) {
            return;
        }

        HeatSolveNodeParams params{};
        if (!tryLoadNodeParams(params)) {
            setStatus("Cannot update flux vector settings for this node.");
            return;
        }

        params.preview.showFluxVectors = checked;
        if (!writeNodeParams(params)) {
            setStatus("Failed to update flux vector settings.");
            return;
        }

        setStatus("Heat settings applied.");
    });
    connect(heatPaletteCheckBox, &QCheckBox::toggled, this, [this](bool checked) {
        if (isSyncing()) {
            return;
        }

        HeatSolveNodeParams params{};
        if (!tryLoadNodeParams(params)) {
            setStatus("Cannot update heat palette settings for this node.");
            return;
        }

        params.preview.showHeatPalette = checked;
        if (!writeNodeParams(params)) {
            setStatus("Failed to update heat palette settings.");
            return;
        }

        setStatus("Heat settings applied.");
    });
    fluxVectorScaleRow->setValueChangedCallback([this](double value) {
        if (isSyncing()) {
            return;
        }

        HeatSolveNodeParams params{};
        if (!tryLoadNodeParams(params)) {
            setStatus("Cannot update flux vector scale for this node.");
            return;
        }

        params.preview.fluxVectorScale = value;
        if (!writeNodeParams(params)) {
            setStatus("Failed to update flux vector scale.");
            return;
        }

        setStatus("Heat settings applied.");
    });
    connect(heatStatusTimer, &QTimer::timeout, this, [this]() {
        updateHeatStatus();
    });
}

void NodeHeatSolverPanel::bind(NodeGraph* graphPtr, const RuntimeQuery* runtimeQueryPtr) {
    runtimeQuery = runtimeQueryPtr;
    NodePanelBase::bind(graphPtr);
}

void NodeHeatSolverPanel::refreshFromNode() {
    HeatSolveNodeParams params{};
    if (!tryLoadNodeParams(params)) {
        setStatus("Failed to read HeatSolve node.");
        return;
    }

    setSyncing(true);
    heatContactThermalConductanceRow->setValue(params.contactThermalConductance);
    heatSimulationDurationRow->setValue(params.simulationDuration);
    heatOverlayCheckBox->setChecked(params.preview.showHeatOverlay);
    fluxVectorsCheckBox->setChecked(params.preview.showFluxVectors);
    heatPaletteCheckBox->setChecked(params.preview.showHeatPalette);
    fluxVectorScaleRow->setValue(params.preview.fluxVectorScale);
    setSyncing(false);

    updateHeatStatus();
}

void NodeHeatSolverPanel::updateHeatStatus() {
    NodeGraphNode node{};
    const bool hasNodeState = bridge() && currentNodeId().isValid() && bridge()->getNode(currentNodeId(), node);
    if (!hasNodeState || getNodeTypeId(node.typeId) != nodegraphtypes::HeatSolve) {
        heatStatusValueLabel->setText("Unavailable");
        return;
    }

    if (!runtimeQuery) {
        const HeatSolveNodeParams params = readHeatSolveNodeParams(node);
        heatStatusValueLabel->setText(params.enabled ? (params.paused ? "Queued Paused" : "Queued") : "Stopped");
        return;
    }

    const bool active = runtimeQuery->isSimulationActive();
    const bool paused = runtimeQuery->isSimulationPaused();

    if (!active) {
        const HeatSolveNodeParams params = readHeatSolveNodeParams(node);
        if (params.enabled) {
            std::string reason;
            if (bridge() && !bridge()->canExecute(reason)) {
                heatStatusValueLabel->setText("Blocked");
                return;
            }
            heatStatusValueLabel->setText(params.paused ? "Pending Pause" : "Pending Start");
            return;
        }
        heatStatusValueLabel->setText("Stopped");
        return;
    }

    if (paused) {
        heatStatusValueLabel->setText("Paused");
        return;
    }

    heatStatusValueLabel->setText("Running");
}

void NodeHeatSolverPanel::startStatusTimer() {
    if (runtimeQuery) {
        heatStatusTimer->start();
    }
}

void NodeHeatSolverPanel::stopStatusTimer() {
    heatStatusTimer->stop();
}

bool NodeHeatSolverPanel::writeSolverSettings() {
    if (!canEdit()) {
        setStatus("Cannot apply solver settings for this node.");
        return false;
    }

    HeatSolveNodeParams params{};
    if (!tryLoadNodeParams(params)) {
        setStatus("Cannot apply solver settings for this node.");
        return false;
    }

    params.contactThermalConductance = heatContactThermalConductanceRow->value();
    params.simulationDuration = heatSimulationDurationRow->value();
    if (!writeNodeParams(params)) {
        setStatus("Failed to update solver settings.");
        return false;
    }

    return true;
}

void NodeHeatSolverPanel::onSolverSettingsEdited() {
    if (isSyncing()) {
        return;
    }

    if (writeSolverSettings()) {
        setStatus("Solver settings applied.");
    }
}

bool NodeHeatSolverPanel::writeNodeParams(const HeatSolveNodeParams& params) {
    if (!canEdit()) {
        return false;
    }

    NodeGraphEditor editor(bridge());
    return writeHeatSolveNodeParams(editor, currentNodeId(), params);
}

bool NodeHeatSolverPanel::tryLoadNodeParams(HeatSolveNodeParams& outParams) const {
    NodeGraphNode node{};
    if (!NodePanelUtils::loadNode(bridge(), currentNodeId(), node)) {
        return false;
    }

    outParams = readHeatSolveNodeParams(node);
    return true;
}
