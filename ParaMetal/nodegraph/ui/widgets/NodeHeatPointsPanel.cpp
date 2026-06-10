#include "NodeHeatPointsPanel.hpp"

#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeHeatPointsParams.hpp"
#include "NodeGraphWidgetStyle.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QVBoxLayout>

NodeHeatPointsPanel::NodeHeatPointsPanel(QWidget* parent)
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

    // Initial Temperature
    QHBoxLayout* initTempRow = new QHBoxLayout();
    initTempRow->addWidget(new QLabel("Initial Temperature (K):", this));
    initialTemperatureEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 10000.0, 2);
    initialTemperatureEdit->setText(QString::number(HeatSimDefaults::ambientTemperature, 'f', 2));
    initTempRow->addWidget(initialTemperatureEdit, 1);
    layout->addLayout(initTempRow);

    // Fixed Temperature
    QHBoxLayout* fixedTempRow = new QHBoxLayout();
    fixedTempRow->addWidget(new QLabel("Fixed Temperature (K):", this));
    fixedTemperatureEdit = nodegraphwidgets::createNumericEdit(this, 0.0, 10000.0, 2);
    fixedTemperatureEdit->setText(QString::number(HeatSimDefaults::ambientTemperature, 'f', 2));
    fixedTempRow->addWidget(fixedTemperatureEdit, 1);
    layout->addLayout(fixedTempRow);

    layout->addStretch();

    connect(boundaryConditionCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this]() {
        applySettings();
    });
    connect(initialTemperatureEdit, &QLineEdit::editingFinished, this, [this]() { applySettings(); });
    connect(fixedTemperatureEdit, &QLineEdit::editingFinished, this, [this]() { applySettings(); });
}

void NodeHeatPointsPanel::applySettings() {
    if (isSyncing() || !canEdit()) {
        return;
    }

    HeatPointsNodeParams params{};
    params.boundaryCondition = static_cast<HeatBoundaryCondition>(boundaryConditionCombo->currentData().toInt());
    params.initialTemperature = initialTemperatureEdit->text().toDouble();
    params.fixedTemperature = fixedTemperatureEdit->text().toDouble();

    NodeGraphEditor editor(bridge());
    if (!writeHeatPointsNodeParams(editor, currentNodeId(), params)) {
        setStatus("Failed to update heat points settings.");
        return;
    }

    setStatus("Heat points settings applied.");
}

void NodeHeatPointsPanel::refreshFromNode() {
    if (!canEdit()) {
        return;
    }

    NodeGraphNode node{};
    if (!loadCurrentNode(node)) {
        return;
    }

    const HeatPointsNodeParams params = readHeatPointsNodeParams(node);
    setSyncing(true);

    for (int i = 0; i < boundaryConditionCombo->count(); ++i) {
        if (boundaryConditionCombo->itemData(i).toInt() == static_cast<int>(params.boundaryCondition)) {
            boundaryConditionCombo->setCurrentIndex(i);
            break;
        }
    }

    initialTemperatureEdit->setText(QString::number(params.initialTemperature, 'f', 2));
    fixedTemperatureEdit->setText(QString::number(params.fixedTemperature, 'f', 2));
    setSyncing(false);
}
