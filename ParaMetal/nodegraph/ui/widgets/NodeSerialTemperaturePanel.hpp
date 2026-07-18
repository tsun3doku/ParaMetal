#pragma once

#include "NodePanelBase.hpp"

class QCheckBox;
class QComboBox;
class QLabel;
class QPushButton;
class QTimer;
class RuntimeQuery;

class NodeSerialTemperaturePanel final : public NodePanelBase {
public:
    explicit NodeSerialTemperaturePanel(QWidget* parent = nullptr);

    void bind(NodeGraph* graph, const RuntimeQuery* runtimeQuery);
    void startStatusTimer();
    void stopStatusTimer();

protected:
    void refreshFromNode() override;

private:
    void refreshPorts();
    void updateStatus();
    void writeSettings();
    uint64_t sourceKey() const;

    const RuntimeQuery* runtimeQuery = nullptr;
    QCheckBox* enabledCheckBox = nullptr;
    QComboBox* portCombo = nullptr;
    QPushButton* refreshButton = nullptr;
    QComboBox* baudCombo = nullptr;
    QLabel* connectionValue = nullptr;
    QLabel* temperatureValue = nullptr;
    QLabel* pollingRateValue = nullptr;
    QTimer* statusTimer = nullptr;
};
