#pragma once

#include "NodePanelBase.hpp"
#include "domain/HeatModelData.hpp"

class QLineEdit;
class QPushButton;
class QComboBox;

class NodeHeatModelPanel final : public NodePanelBase {
public:
    explicit NodeHeatModelPanel(QWidget* parent = nullptr);

protected:
    void refreshFromNode() override;

private:
    void applySettings();

    QComboBox* boundaryConditionCombo = nullptr;
    QLineEdit* temperatureEdit = nullptr;
    QLineEdit* densityEdit = nullptr;
    QLineEdit* specificHeatEdit = nullptr;
    QLineEdit* conductivityEdit = nullptr;
    QPushButton* applyButton = nullptr;
};