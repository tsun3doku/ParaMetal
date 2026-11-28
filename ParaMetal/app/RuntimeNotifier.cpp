#include "RuntimeNotifier.hpp"

#include <cmath>

void RuntimeNotifier::onTemperatures(const TelemetryTemperatures& telemetry) {
    if (!telemetry.values && telemetry.count != 0) {
        return;
    }

    const int nodeId = static_cast<int>(telemetry.nodeId);
    const qsizetype count = static_cast<qsizetype>(telemetry.count);
    bool changed = false;
    QList<float> qtemps;

    {
        const QMutexLocker lock(&statusMutex);
        auto it = heatProbeTemperaturesList.find(nodeId);
        if (it == heatProbeTemperaturesList.end() || it->second.size() != count) {
            changed = true;
        } else {
            for (qsizetype i = 0; i < count; ++i) {
                if (std::abs(it->second[i] - telemetry.values[i]) > 0.001f) {
                    changed = true;
                    break;
                }
            }
        }

        if (changed) {
            qtemps.reserve(count);
            for (qsizetype i = 0; i < count; ++i) {
                qtemps.append(telemetry.values[i]);
            }
            heatProbeTemperaturesList[nodeId] = qtemps;
        }
    }

    if (changed) {
        emit heatProbeTemperaturesChanged(nodeId, qtemps);
    }
}

void RuntimeNotifier::onConnection(const TelemetryConnection& telemetry) {
    if (!telemetry.status) {
        return;
    }

    const int nodeId = static_cast<int>(telemetry.nodeId);
    const SerialTemperatureRuntime::Status& status = *telemetry.status;

    QString connectionText;
    switch (status.connection) {
    case SerialTemperatureRuntime::ConnectionState::Error:
        connectionText = QStringLiteral("Error: ") + QString::fromStdString(status.detail);
        break;
    case SerialTemperatureRuntime::ConnectionState::DisconnectedHolding:
        connectionText = QStringLiteral("Disconnected - holding last reading: ") +
            QString::fromStdString(status.detail);
        break;
    default:
        connectionText = QString::fromStdString(status.detail);
        break;
    }

    QString temperatureText = QStringLiteral("--");
    QString pollingRateText = QStringLiteral("--");
    if (status.latestReading) {
        temperatureText = QString::number(status.latestReading->temperatureC, 'f', 2) +
            QStringLiteral(" \u00B0C");
        pollingRateText = QString::number(status.latestReading->pollingRateHz, 'f', 1) +
            QStringLiteral(" Hz");
    }

    publishSerialStatus(nodeId, connectionText, temperatureText, pollingRateText);
}
