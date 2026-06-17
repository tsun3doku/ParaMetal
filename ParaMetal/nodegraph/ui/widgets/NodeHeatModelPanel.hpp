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
    bool writeParameters();
    void onPresetEdited();
    void onSettingsEdited();
    void onMaterialValueEdited();
    void setMaterialPresetCombo(HeatMaterialPresetId presetId);

    QComboBox* materialPresetCombo = nullptr;
    QComboBox* boundaryConditionCombo = nullptr;
    QLineEdit* temperatureEdit = nullptr;
    QLineEdit* densityEdit = nullptr;
    QLineEdit* specificHeatEdit = nullptr;
    QLineEdit* conductivityEdit = nullptr;
};
