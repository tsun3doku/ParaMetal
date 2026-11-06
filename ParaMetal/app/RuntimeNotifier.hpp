#pragma once

#include "UiRuntimeTypes.hpp"

#include <QMutex>
#include <QObject>
#include <QString>
#include <QVariantList>
#include <cmath>

class RuntimeNotifier final : public QObject {
    Q_OBJECT

public:
    explicit RuntimeNotifier(QObject* parent = nullptr) : QObject(parent) {}

    Q_INVOKABLE bool heatSolveActive() const {
        const QMutexLocker lock(&statusMutex);
        return heatSolveActiveState;
    }
    Q_INVOKABLE bool heatSolvePaused() const {
        const QMutexLocker lock(&statusMutex);
        return heatSolvePausedState;
    }
    Q_INVOKABLE QString serialConnectionText(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        const auto it = serialStatusMap.find(nodeId);
        return it != serialStatusMap.end() ? it->second.connection : QStringLiteral("Not used by an active Heat Solve");
    }
    Q_INVOKABLE QString serialTemperatureText(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        const auto it = serialStatusMap.find(nodeId);
        return it != serialStatusMap.end() ? it->second.temperature : QStringLiteral("--");
    }
    Q_INVOKABLE QString serialPollingRateText(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        const auto it = serialStatusMap.find(nodeId);
        return it != serialStatusMap.end() ? it->second.pollingRate : QStringLiteral("--");
    }
    Q_INVOKABLE QString heatSolveError() const {
        const QMutexLocker lock(&statusMutex);
        return heatSolveErrorState;
    }

    Q_INVOKABLE QList<float> heatProbeTemperatures(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        auto it = heatProbeTemperaturesList.find(nodeId);
        return it != heatProbeTemperaturesList.end() ? it->second : QList<float>{};
    }

    Q_INVOKABLE QString exportStatusMessage(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        auto it = exportStatusMap.find(nodeId);
        return it != exportStatusMap.end() ? it->second.message : QString{};
    }

    Q_INVOKABLE bool exportStatusSuccess(int nodeId) const {
        const QMutexLocker lock(&statusMutex);
        auto it = exportStatusMap.find(nodeId);
        return it != exportStatusMap.end() ? it->second.success : false;
    }

    Q_INVOKABLE void clearExportStatus(int nodeId) {
        {
            const QMutexLocker lock(&statusMutex);
            exportStatusMap.erase(nodeId);
        }
        emit exportStatusChanged(nodeId, true, QString{});
    }

    // Node ids are reused across graphs, so per-node UI state must be dropped
    // whenever the graph identity changes.
    void resetNodeStatuses() {
        std::vector<int> clearedExportIds;
        {
            const QMutexLocker lock(&statusMutex);
            clearedExportIds.reserve(exportStatusMap.size());
            for (const auto& entry : exportStatusMap) clearedExportIds.push_back(entry.first);
            exportStatusMap.clear();
            heatProbeTemperaturesList.clear();
            serialStatusMap.clear();
        }
        for (int nodeId : clearedExportIds) {
            emit exportStatusChanged(nodeId, true, QString{});
        }
    }

    void publishExportStatus(int nodeId, bool success, const QString& message) {
        {
            const QMutexLocker lock(&statusMutex);
            exportStatusMap[nodeId] = {success, message};
        }
        emit exportStatusChanged(nodeId, success, message);
    }

    void publishExportProducts(int nodeId, qulonglong requestId, qulonglong graphRevision,
                               const QVariantList& products, const QString& error) {
        emit exportProductsReady(nodeId, requestId, graphRevision, products, error);
    }

    void publishHeatProbeTemperatures(int nodeId, const std::vector<float>& temps) {
        bool changed = false;
        QList<float> qtemps;
        qtemps.reserve(static_cast<qsizetype>(temps.size()));
        for (float t : temps) qtemps.append(t);

        {
            const QMutexLocker lock(&statusMutex);
            auto it = heatProbeTemperaturesList.find(nodeId);
            if (it == heatProbeTemperaturesList.end() || it->second.size() != qtemps.size()) {
                heatProbeTemperaturesList[nodeId] = qtemps;
                changed = true;
            } else {
                for (qsizetype i = 0; i < qtemps.size(); ++i) {
                    if (std::abs(it->second[i] - qtemps[i]) > 0.001f) {
                        it->second = qtemps;
                        changed = true;
                        break;
                    }
                }
            }
        }
        if (changed) {
            emit heatProbeTemperaturesChanged(nodeId, qtemps);
        }
    }

    void notifyProbePickingCompleted(int nodeId) {
        emit probePickingCompleted(nodeId);
    }

    void notifyGraphSelectionChanged(NodeGraphNodeId nodeId) {
        emit graphSelectionChanged(nodeId);
    }

    void publishHeatSolveStatus(bool active, bool paused) {
        {
            const QMutexLocker lock(&statusMutex);
            if (heatSolveActiveState == active && heatSolvePausedState == paused) return;
            heatSolveActiveState = active;
            heatSolvePausedState = paused;
        }
        emit heatSolveStatusChanged(active, paused);
    }

    void publishHeatSolveError(const QString& error) {
        {
            const QMutexLocker lock(&statusMutex);
            if (heatSolveErrorState == error) return;
            heatSolveErrorState = error;
        }
        emit heatSolveErrorChanged(error);
    }

    void publishSerialStatus(int nodeId, const QString& connection, const QString& temperature, const QString& pollingRate) {
        {
            const QMutexLocker lock(&statusMutex);
            SerialStatusInfo& status = serialStatusMap[nodeId];
            if (status.connection == connection &&
                status.temperature == temperature &&
                status.pollingRate == pollingRate) return;
            status = {connection, temperature, pollingRate};
        }
        emit serialStatusChanged(nodeId, connection, temperature, pollingRate);
    }

signals:
    void viewportStateChanged(const ViewportUiState& state);
    void heatPaletteVisibilityChanged(bool visible);
    void timelineStateChanged(const TimelineUiState& state);
    void heatSolveStatusChanged(bool active, bool paused);
    void heatSolveErrorChanged(const QString& error);
    void serialStatusChanged(int nodeId, const QString& connection, const QString& temperature, const QString& pollingRate);
    void heatProbeTemperaturesChanged(int nodeId, const QList<float>& temperatures);
    void probePickingCompleted(int nodeId);
    void graphSelectionChanged(NodeGraphNodeId nodeId);
    void exportStatusChanged(int nodeId, bool success, const QString& message);
    void exportProductsReady(int nodeId, qulonglong requestId, qulonglong graphRevision,
                             const QVariantList& products, const QString& error);
    void nodeParametersRequested(
        NodeGraphNodeId nodeId,
        const std::vector<NodeGraphParamValue>& parameters);

private:
    struct ExportStatusInfo {
        bool success = false;
        QString message;
    };

    struct SerialStatusInfo {
        QString connection;
        QString temperature;
        QString pollingRate;
    };

    mutable QMutex statusMutex;
    bool heatSolveActiveState = false;
    bool heatSolvePausedState = false;
    QString heatSolveErrorState;
    std::unordered_map<int, QList<float>> heatProbeTemperaturesList;
    std::unordered_map<int, ExportStatusInfo> exportStatusMap;
    std::unordered_map<int, SerialStatusInfo> serialStatusMap;
};
