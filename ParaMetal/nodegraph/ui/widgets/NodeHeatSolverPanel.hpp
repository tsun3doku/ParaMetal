#pragma once

#include "NodePanelBase.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"

class QCheckBox;
class QLabel;
class QTimer;
class NodeGraphSliderRow;
class RuntimeQuery;

class NodeHeatSolverPanel final : public NodePanelBase {
public:
    explicit NodeHeatSolverPanel(QWidget* parent = nullptr);

    void bind(NodeGraphBridge* nodeGraphBridge, const RuntimeQuery* runtimeQuery);

    void updateHeatStatus();

    void startStatusTimer();
    void stopStatusTimer();

protected:
    void refreshFromNode() override;

private:
    bool writeNodeParams(const HeatSolveNodeParams& params);
    bool tryLoadNodeParams(HeatSolveNodeParams& outParams) const;
    bool writeSolverSettings();
    void onSolverSettingsEdited();

    const RuntimeQuery* runtimeQuery = nullptr;

    QLabel* heatStatusValueLabel = nullptr;
    NodeGraphSliderRow* heatContactThermalConductanceRow = nullptr;
    NodeGraphSliderRow* heatSimulationDurationRow = nullptr;
    QCheckBox* heatOverlayCheckBox = nullptr;
    QCheckBox* fluxVectorsCheckBox = nullptr;
    QCheckBox* heatPaletteCheckBox = nullptr;
    NodeGraphSliderRow* fluxVectorScaleRow = nullptr;
    QTimer* heatStatusTimer = nullptr;
};
