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
    const bool hasReading = status.latestReading.has_value();
    const uint64_t revision = hasReading ? status.latestReading->revision : 0;
    const float readingTemperatureC = hasReading ? status.latestReading->temperatureC : 0.0f;
    const float readingPollingRateHz = hasReading ? status.latestReading->pollingRateHz : 0.0f;

    {
        const QMutexLocker lock(&statusMutex);
        SerialStatusInfo& cached = serialStatusMap[nodeId];
        if (cached.initialized &&
            cached.connectionState == status.connection &&
            cached.detail == status.detail &&
            cached.hasReading == hasReading &&
            cached.readingRevision == revision &&
            cached.readingTemperatureC == readingTemperatureC &&
            cached.readingPollingRateHz == readingPollingRateHz) {
            return;
        }
        cached.initialized = true;
        cached.connectionState = status.connection;
        cached.detail = status.detail;
        cached.hasReading = hasReading;
        cached.readingRevision = revision;
        cached.readingTemperatureC = readingTemperatureC;
        cached.readingPollingRateHz = readingPollingRateHz;
    }

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
