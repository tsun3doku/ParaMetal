#pragma once

#include "NodePanelBase.hpp"
#include "domain/HeatModelData.hpp"

class QLineEdit;
class QComboBox;

class NodeHeatPointsPanel final : public NodePanelBase {
public:
    explicit NodeHeatPointsPanel(QWidget* parent = nullptr);

protected:
    void refreshFromNode() override;

private:
    void applySettings();

    QComboBox* boundaryConditionCombo = nullptr;
    QLineEdit* initialTemperatureEdit = nullptr;
    QLineEdit* fixedTemperatureEdit = nullptr;
};
