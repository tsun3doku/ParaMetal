#include "NodeSerialTemperaturePanel.hpp"

#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphEditor.hpp"
#include "nodegraph/NodeSerialTemperatureParams.hpp"
#include "runtime/RuntimeInterfaces.hpp"
#include "serial/SerialPort.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

namespace {
QString connectionText(SerialTemperatureRuntime::ConnectionState state) {
    using State = SerialTemperatureRuntime::ConnectionState;
    switch (state) {
    case State::Disabled: return "Disabled";
    case State::NoPortSelected: return "No port selected";
    case State::Connecting: return "Connecting";
    case State::WaitingForReading: return "Waiting for reading";
    case State::Live: return "Live";
    case State::StaleHolding: return "Stale - holding last reading";
    case State::DisconnectedHolding: return "Disconnected - holding last reading";
    case State::Error: return "Error";
    }
    return "Unknown";
}
}

NodeSerialTemperaturePanel::NodeSerialTemperaturePanel(QWidget* parent)
    : NodePanelBase(parent) {
    QVBoxLayout* layout = static_cast<QVBoxLayout*>(this->layout());

    enabledCheckBox = new QCheckBox("Enabled", this);
    layout->addWidget(enabledCheckBox);

    QHBoxLayout* portRow = new QHBoxLayout();
    portRow->addWidget(new QLabel("Port:", this));
    portCombo = new QComboBox(this);
    portRow->addWidget(portCombo, 1);
    refreshButton = new QPushButton("Refresh", this);
    portRow->addWidget(refreshButton);
    layout->addLayout(portRow);

    QHBoxLayout* baudRow = new QHBoxLayout();
    baudRow->addWidget(new QLabel("Baud Rate:", this));
    baudCombo = new QComboBox(this);
    for (uint32_t baud : {9600u, 19200u, 38400u, 57600u, 115200u, 230400u}) {
        baudCombo->addItem(QString::number(baud), baud);
    }
    baudRow->addWidget(baudCombo, 1);
    layout->addLayout(baudRow);

    QHBoxLayout* connectionRow = new QHBoxLayout();
    connectionRow->addWidget(new QLabel("Status:", this));
    connectionValue = new QLabel("Unavailable", this);
    connectionValue->setWordWrap(true);
    connectionRow->addWidget(connectionValue, 1);
    layout->addLayout(connectionRow);

    QHBoxLayout* temperatureRow = new QHBoxLayout();
    temperatureRow->addWidget(new QLabel("Latest Temperature:", this));
    temperatureValue = new QLabel("--", this);
    temperatureRow->addWidget(temperatureValue, 1);
    layout->addLayout(temperatureRow);

    QHBoxLayout* pollingRateRow = new QHBoxLayout();
    pollingRateRow->addWidget(new QLabel("Polling Rate:", this));
    pollingRateValue = new QLabel("--", this);
    pollingRateRow->addWidget(pollingRateValue, 1);
    layout->addLayout(pollingRateRow);
    layout->addStretch();

    statusTimer = new QTimer(this);
    statusTimer->setInterval(200);
    connect(statusTimer, &QTimer::timeout, this, [this]() { updateStatus(); });
    connect(refreshButton, &QPushButton::clicked, this, [this]() { refreshPorts(); });
    connect(enabledCheckBox, &QCheckBox::toggled, this, [this](bool) { writeSettings(); });
    connect(portCombo, &QComboBox::currentTextChanged, this, [this](const QString&) { writeSettings(); });
    connect(baudCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) { writeSettings(); });
}

void NodeSerialTemperaturePanel::bind(NodeGraph* graph, const RuntimeQuery* query) {
    runtimeQuery = query;
    NodePanelBase::bind(graph);
}

void NodeSerialTemperaturePanel::startStatusTimer() {
    if (runtimeQuery) statusTimer->start();
    updateStatus();
}

void NodeSerialTemperaturePanel::stopStatusTimer() { statusTimer->stop(); }

void NodeSerialTemperaturePanel::refreshFromNode() {
    NodeGraphNode node{};
    if (!loadCurrentNode(node)) return;
    const SerialTemperatureNodeParams params = readSerialTemperatureNodeParams(node);
    setSyncing(true);
    enabledCheckBox->setChecked(params.enabled);
    refreshPorts();
    const QString configuredPort = QString::fromStdString(params.portName);
    int portIndex = portCombo->findData(configuredPort);
    if (portIndex < 0 && !configuredPort.isEmpty()) {
        portCombo->addItem(configuredPort + " (unavailable)", configuredPort);
        portIndex = portCombo->count() - 1;
    }
    portCombo->setCurrentIndex(portIndex);
    int baudIndex = baudCombo->findData(params.baudRate);
    if (baudIndex < 0) {
        baudCombo->addItem(QString::number(params.baudRate), params.baudRate);
        baudIndex = baudCombo->count() - 1;
    }
    baudCombo->setCurrentIndex(baudIndex);
    setSyncing(false);
    updateStatus();
}

void NodeSerialTemperaturePanel::refreshPorts() {
    const QString selected = portCombo->currentText();
    portCombo->blockSignals(true);
    portCombo->clear();
    for (const SerialPortInfo& port : SerialPort::enumeratePorts()) {
        portCombo->addItem(QString::fromStdString(port.displayName), QString::fromStdString(port.portName));
    }
    int index = portCombo->findData(selected);
    if (index >= 0) portCombo->setCurrentIndex(index);
    else if (!selected.isEmpty()) {
        portCombo->addItem(selected + " (unavailable)", selected);
        portCombo->setCurrentIndex(portCombo->count() - 1);
    }
    portCombo->blockSignals(false);
}

uint64_t NodeSerialTemperaturePanel::sourceKey() const {
    NodeGraphNode node{};
    if (!bridge() || !bridge()->getNode(currentNodeId(), node) || node.outputs.empty()) return 0;
    return NodeSocketKey(node.id, node.outputs.front().id).value;
}

void NodeSerialTemperaturePanel::updateStatus() {
    SerialTemperatureRuntime::Status status{};
    if (!runtimeQuery || !runtimeQuery->getSerialTemperatureStatus(sourceKey(), status)) {
        connectionValue->setText("Not used by an active Heat Solve");
        temperatureValue->setText("--");
        pollingRateValue->setText("--");
        return;
    }
    QString statusText = connectionText(status.connection);
    const QString detail = QString::fromStdString(status.detail);
    if (!detail.isEmpty() && detail != statusText) {
        statusText += ": " + detail;
    }
    connectionValue->setText(statusText);
    if (!status.latestReading) {
        temperatureValue->setText("--");
        pollingRateValue->setText("--");
        return;
    }
    temperatureValue->setText(QString::number(status.latestReading->temperatureC, 'f', 2) + " C");
    pollingRateValue->setText(status.latestReading->pollingRateHz > 0.0f
        ? QString::number(status.latestReading->pollingRateHz, 'f', 2) + " Hz"
        : "Estimating...");
}

void NodeSerialTemperaturePanel::writeSettings() {
    if (isSyncing() || !canEdit()) return;
    SerialTemperatureNodeParams params{};
    params.enabled = enabledCheckBox->isChecked();
    params.portName = portCombo->currentData().toString().toStdString();
    params.baudRate = baudCombo->currentData().toUInt();
    NodeGraphEditor editor(bridge());
    if (writeSerialTemperatureNodeParams(editor, currentNodeId(), params)) {
        setStatus("Serial temperature settings applied.");
    } else {
        setStatus("Failed to apply serial temperature settings.");
    }
}
