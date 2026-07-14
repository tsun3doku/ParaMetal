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
    boundaryConditionCombo->addItem("Adiabatic", static_cast<int>(BoundaryCondition::Type::Adiabatic));
    boundaryConditionCombo->addItem("Dirichlet Temperature", static_cast<int>(BoundaryCondition::Type::DirichletTemperature));
    boundaryConditionCombo->addItem("Neumann Heat Flux", static_cast<int>(BoundaryCondition::Type::NeumannHeatFlux));
    boundaryConditionCombo->addItem("Robin Convection", static_cast<int>(BoundaryCondition::Type::RobinConvection));
    bcRow->addWidget(boundaryConditionCombo, 1);
    layout->addLayout(bcRow);

    QHBoxLayout* initialTempRow = new QHBoxLayout();
    initialTempRow->addWidget(new QLabel("Initial Temperature (C):", this));
    initialTemperatureEdit = nodegraphwidgets::createNumericEdit(this, -273.15, 10000.0, 2);
    initialTemperatureEdit->setText(QString::number(HeatSimDefaults::ambientTemperatureC, 'f', 2));
    initialTempRow->addWidget(initialTemperatureEdit, 1);
    layout->addLayout(initialTempRow);

    QHBoxLayout* boundaryTempRow = new QHBoxLayout();
    boundaryTempRow->addWidget(new QLabel("Boundary Temperature (C):", this));
    boundaryTemperatureEdit = nodegraphwidgets::createNumericEdit(this, -273.15, 10000.0, 2);
    boundaryTemperatureEdit->setText(QString::number(HeatSimDefaults::ambientTemperatureC, 'f', 2));
    boundaryTempRow->addWidget(boundaryTemperatureEdit, 1);
    layout->addLayout(boundaryTempRow);

    QHBoxLayout* heatFluxRow = new QHBoxLayout();
    heatFluxRow->addWidget(new QLabel("Inward Heat Flux (W/m²):", this));
    heatFluxEdit = nodegraphwidgets::createNumericEdit(this, -1.0e12, 1.0e12, 3);
    heatFluxEdit->setText("0");
    heatFluxRow->addWidget(heatFluxEdit, 1);
    layout->addLayout(heatFluxRow);

    QHBoxLayout* htcRow = new QHBoxLayout();
    htcRow->addWidget(new QLabel("HTC (W/(m²·K)):", this));
    heatTransferCoefficientEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 1.0e12, 3);
    heatTransferCoefficientEdit->setText("0");
    htcRow->addWidget(heatTransferCoefficientEdit, 1);
    layout->addLayout(htcRow);

    QHBoxLayout* volumetricRow = new QHBoxLayout();
    volumetricRow->addWidget(new QLabel("Power Density (W/m³):", this));
    volumetricPowerDensityEdit = nodegraphwidgets::createNumericEdit(this, -1.0e15, 1.0e15, 3);
    volumetricPowerDensityEdit->setText("0");
    volumetricRow->addWidget(volumetricPowerDensityEdit, 1);
    layout->addLayout(volumetricRow);

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
    connect(initialTemperatureEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
    connect(boundaryTemperatureEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
    connect(heatFluxEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
    connect(heatTransferCoefficientEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
    connect(volumetricPowerDensityEdit, &QLineEdit::editingFinished, this, [this]() { onSettingsEdited(); });
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
    params.boundaryConditionType = static_cast<BoundaryCondition::Type>(boundaryConditionCombo->currentData().toInt());
    params.boundaryTemperatureC = boundaryTemperatureEdit->text().toDouble();
    params.heatFlux = heatFluxEdit->text().toDouble();
    params.heatTransferCoefficient = heatTransferCoefficientEdit->text().toDouble();
    params.volumetricPowerDensity = volumetricPowerDensityEdit->text().toDouble();
    params.density = densityEdit->text().toDouble();
    params.specificHeat = specificHeatEdit->text().toDouble();
    params.conductivity = conductivityEdit->text().toDouble();
    params.initialTemperatureC = initialTemperatureEdit->text().toDouble();
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
        if (boundaryConditionCombo->itemData(i).toInt() == static_cast<int>(params.boundaryConditionType)) {
            boundaryConditionCombo->setCurrentIndex(i);
            break;
        }
    }

    initialTemperatureEdit->setText(QString::number(params.initialTemperatureC, 'f', 2));
    boundaryTemperatureEdit->setText(QString::number(params.boundaryTemperatureC, 'f', 2));
    heatFluxEdit->setText(QString::number(params.heatFlux, 'g', 8));
    heatTransferCoefficientEdit->setText(QString::number(params.heatTransferCoefficient, 'g', 8));
    volumetricPowerDensityEdit->setText(QString::number(params.volumetricPowerDensity, 'g', 8));
    densityEdit->setText(QString::number(params.density, 'f', 1));
    specificHeatEdit->setText(QString::number(params.specificHeat, 'f', 1));
    conductivityEdit->setText(QString::number(params.conductivity, 'f', 2));
    setSyncing(false);
}
