#include "NodeHeatModelPanel.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"

#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeHeatModelParams.hpp"
#include "NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QComboBox>

NodeHeatModelPanel::NodeHeatModelPanel(QWidget* parent)
    : NodePanelBase(parent) {
    QVBoxLayout* layout = static_cast<QVBoxLayout*>(this->layout());

    // Boundary Condition
    QHBoxLayout* bcRow = new QHBoxLayout();
    bcRow->addWidget(new QLabel("Boundary Condition:", this));
    boundaryConditionCombo = new QComboBox(this);
    nodegraphwidgets::styleComboBox(boundaryConditionCombo);
    boundaryConditionCombo->addItem("None", static_cast<int>(HeatBoundaryCondition::None));
    boundaryConditionCombo->addItem("Fixed Temperature", static_cast<int>(HeatBoundaryCondition::FixedTemperature));
    boundaryConditionCombo->addItem("Fixed Power", static_cast<int>(HeatBoundaryCondition::FixedPower));
    bcRow->addWidget(boundaryConditionCombo, 1);
    layout->addLayout(bcRow);

    // Temperature (for Fixed Temperature BC)
    QHBoxLayout* tempRow = new QHBoxLayout();
    tempRow->addWidget(new QLabel("Fixed Temperature (K):", this));
    temperatureEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 10000.0, 2);
    temperatureEdit->setText(QString::number(HeatSimDefaults::ambientTemperature, 'f', 2));
    tempRow->addWidget(temperatureEdit, 1);
    layout->addLayout(tempRow);

    // Density
    QHBoxLayout* densityRow = new QHBoxLayout();
    densityRow->addWidget(new QLabel("Density (kg/m³):", this));
    densityEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 100000.0, 1);
    densityEdit->setText(QString::number(HeatSimDefaults::density, 'f', 1));
    densityRow->addWidget(densityEdit, 1);
    layout->addLayout(densityRow);

    // Specific Heat
    QHBoxLayout* specificHeatRow = new QHBoxLayout();
    specificHeatRow->addWidget(new QLabel("Specific Heat (J/(kg·K)):", this));
    specificHeatEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 100000.0, 1);
    specificHeatEdit->setText(QString::number(HeatSimDefaults::specificHeat, 'f', 1));
    specificHeatRow->addWidget(specificHeatEdit, 1);
    layout->addLayout(specificHeatRow);

    // Conductivity
    QHBoxLayout* conductivityRow = new QHBoxLayout();
    conductivityRow->addWidget(new QLabel("Conductivity (W/(m·K)):", this));
    conductivityEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 10000.0, 2);
    conductivityEdit->setText(QString::number(HeatSimDefaults::conductivity, 'f', 2));
    conductivityRow->addWidget(conductivityEdit, 1);
    layout->addLayout(conductivityRow);

    QHBoxLayout* actionRow = new QHBoxLayout();
    applyButton = new QPushButton("Apply Settings", this);
    actionRow->addWidget(applyButton);
    layout->addLayout(actionRow);

    layout->addStretch();

    connect(applyButton, &QPushButton::clicked, this, [this]() {
        applySettings();
    });
}

void NodeHeatModelPanel::applySettings() {
    if (!canEdit()) {
        setStatus("Cannot apply settings for this node.");
        return;
    }

    NodeGraphEditor editor(bridge());
    HeatModelNodeParams params{};
    params.boundaryCondition = static_cast<HeatBoundaryCondition>(boundaryConditionCombo->currentData().toInt());
    params.fixedTemperatureValue = temperatureEdit->text().toDouble();
    params.density = densityEdit->text().toDouble();
    params.specificHeat = specificHeatEdit->text().toDouble();
    params.conductivity = conductivityEdit->text().toDouble();
    params.initialTemperature = temperatureEdit->text().toDouble();
    if (!writeHeatModelNodeParams(editor, currentNodeId(), params)) {
        setStatus("Failed to update heat model settings.");
        return;
    }

    setStatus("Heat model settings applied.");
}

void NodeHeatModelPanel::refreshFromNode() {
    if (!canEdit()) {
        return;
    }

    NodeGraphNode node{};
    if (!loadCurrentNode(node)) {
        return;
    }

    const HeatModelNodeParams params = readHeatModelNodeParams(node);

    // Set boundary condition combo
    for (int i = 0; i < boundaryConditionCombo->count(); ++i) {
        if (boundaryConditionCombo->itemData(i).toInt() == static_cast<int>(params.boundaryCondition)) {
            boundaryConditionCombo->setCurrentIndex(i);
            break;
        }
    }

    temperatureEdit->setText(QString::number(params.fixedTemperatureValue, 'f', 2));
    densityEdit->setText(QString::number(params.density, 'f', 1));
    specificHeatEdit->setText(QString::number(params.specificHeat, 'f', 1));
    conductivityEdit->setText(QString::number(params.conductivity, 'f', 2));
}
