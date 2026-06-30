#include "NodeHeatModelPanel.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"

#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeHeatMaterialPresets.hpp"
#include "nodegraph/NodeHeatModelParams.hpp"
#include "NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QComboBox>

NodeHeatModelPanel::NodeHeatModelPanel(QWidget* parent)
    : NodePanelBase(parent) {
    QVBoxLayout* layout = static_cast<QVBoxLayout*>(this->layout());

    QHBoxLayout* presetRow = new QHBoxLayout();
    presetRow->addWidget(new QLabel("Material Preset:", this));
    materialPresetCombo = new QComboBox(this);
    nodegraphwidgets::styleComboBox(materialPresetCombo);
    materialPresetCombo->addItem("Aluminum", static_cast<int>(HeatMaterialPresetId::Aluminum));
    materialPresetCombo->addItem("Copper", static_cast<int>(HeatMaterialPresetId::Copper));
    materialPresetCombo->addItem("Iron", static_cast<int>(HeatMaterialPresetId::Iron));
    materialPresetCombo->addItem("Ceramic", static_cast<int>(HeatMaterialPresetId::Ceramic));
    materialPresetCombo->addItem("Custom", static_cast<int>(HeatMaterialPresetId::Custom));
    presetRow->addWidget(materialPresetCombo, 1);
    layout->addLayout(presetRow);

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

    layout->addStretch();

    connect(materialPresetCombo, &QComboBox::currentIndexChanged, this, [this](int) { onPresetEdited(); });
    connect(boundaryConditionCombo, &QComboBox::currentIndexChanged, this, [this](int) { onSettingsEdited(); });
    connect(temperatureEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
    connect(densityEdit, &QLineEdit::editingFinished, this, [this]() { onMaterialValueEdited(); });
    connect(specificHeatEdit, &QLineEdit::editingFinished, this, [this]() { onMaterialValueEdited(); });
    connect(conductivityEdit, &QLineEdit::editingFinished, this, [this]() { onMaterialValueEdited(); });
}

bool NodeHeatModelPanel::writeParameters() {
    if (!canEdit()) {
        setStatus("Cannot apply settings for this node.");
        return false;
    }

    NodeGraphEditor editor(bridge());
    HeatModelNodeParams params{};
    params.materialPreset = static_cast<HeatMaterialPresetId>(materialPresetCombo->currentData().toInt());
    params.boundaryCondition = static_cast<HeatBoundaryCondition>(boundaryConditionCombo->currentData().toInt());
    params.fixedTemperatureValue = temperatureEdit->text().toDouble();
    params.density = densityEdit->text().toDouble();
    params.specificHeat = specificHeatEdit->text().toDouble();
    params.conductivity = conductivityEdit->text().toDouble();
    params.initialTemperature = temperatureEdit->text().toDouble();
    if (!writeHeatModelNodeParams(editor, currentNodeId(), params)) {
        setStatus("Failed to update heat model settings.");
        return false;
    }

    return true;
}

void NodeHeatModelPanel::onPresetEdited() {
    if (isSyncing()) {
        return;
    }

    const auto presetId = static_cast<HeatMaterialPresetId>(materialPresetCombo->currentData().toInt());
    if (presetId != HeatMaterialPresetId::Custom) {
        const HeatMaterialPreset& preset = heatMaterialPresetById(presetId);
        setSyncing(true);
        densityEdit->setText(QString::number(preset.density, 'f', 1));
        specificHeatEdit->setText(QString::number(preset.specificHeat, 'f', 1));
        conductivityEdit->setText(QString::number(preset.conductivity, 'f', 2));
        setSyncing(false);
    }

    if (writeParameters()) {
        setStatus("Heat model settings applied.");
    }
}

void NodeHeatModelPanel::onSettingsEdited() {
    if (isSyncing()) {
        return;
    }

    if (writeParameters()) {
        setStatus("Heat model settings applied.");
    }
}

void NodeHeatModelPanel::onMaterialValueEdited() {
    if (isSyncing()) {
        return;
    }

    setSyncing(true);
    setMaterialPresetCombo(HeatMaterialPresetId::Custom);
    setSyncing(false);

    if (writeParameters()) {
        setStatus("Heat model settings applied.");
    }
}

void NodeHeatModelPanel::setMaterialPresetCombo(HeatMaterialPresetId presetId) {
    const int presetValue = static_cast<int>(presetId);
    for (int i = 0; i < materialPresetCombo->count(); ++i) {
        if (materialPresetCombo->itemData(i).toInt() == presetValue) {
            materialPresetCombo->setCurrentIndex(i);
            return;
        }
    }
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

    setSyncing(true);
    setMaterialPresetCombo(params.materialPreset);

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
    setSyncing(false);
}
