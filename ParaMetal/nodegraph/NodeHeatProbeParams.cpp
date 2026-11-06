#include "NodeHeatProbeParams.hpp"

#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>

std::string serializeHeatProbesJson(const std::vector<HeatProbeItem>& items) {
    QJsonArray array;
    for (const auto& item : items) {
        QJsonObject obj;
        obj["id"] = static_cast<int>(item.id);
        obj["name"] = QString::fromStdString(item.name);
        obj["x"] = static_cast<double>(item.worldPosition.x);
        obj["y"] = static_cast<double>(item.worldPosition.y);
        obj["z"] = static_cast<double>(item.worldPosition.z);
        obj["enabled"] = item.enabled;
        array.append(obj);
    }
    return QString::fromUtf8(QJsonDocument(array).toJson(QJsonDocument::Compact)).toStdString();
}

std::vector<HeatProbeItem> deserializeHeatProbesJson(const std::string& jsonStr) {
    std::vector<HeatProbeItem> items;
    if (jsonStr.empty()) return items;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(jsonStr), &err);
    if (err.error != QJsonParseError::NoError || !doc.isArray()) return items;

    const QJsonArray array = doc.array();
    uint32_t fallbackId = 1;
    for (const auto& val : array) {
        if (!val.isObject()) continue;
        const QJsonObject obj = val.toObject();
        HeatProbeItem item{};
        item.id = obj.contains("id") ? static_cast<uint32_t>(obj["id"].toInt(fallbackId)) : fallbackId;
        item.name = obj.contains("name") ? obj["name"].toString().toStdString() : ("Probe_" + std::to_string(item.id));
        item.worldPosition.x = static_cast<float>(obj["x"].toDouble(0.0));
        item.worldPosition.y = static_cast<float>(obj["y"].toDouble(0.0));
        item.worldPosition.z = static_cast<float>(obj["z"].toDouble(0.0));
        item.enabled = obj.contains("enabled") ? obj["enabled"].toBool(true) : true;
        items.push_back(item);
        fallbackId = std::max(fallbackId, item.id + 1);
    }
    return items;
}

HeatProbeNodeParams readHeatProbeNodeParams(const NodeGraphNode& node) {
    HeatProbeNodeParams params{};
    params.enabled = NodeParamUtils::readBoolParam(node, nodegraphparams::heatprobe::Enabled, true);
    const int opcPort = NodeParamUtils::readIntParam(node, nodegraphparams::heatprobe::OpcPort, 4840);
    params.opcPort = (opcPort > 0) ? static_cast<uint32_t>(opcPort) : 4840;

    const std::string probesJson = NodeParamUtils::readStringParam(node, nodegraphparams::heatprobe::ProbesJson);
    if (!probesJson.empty()) {
        params.items = deserializeHeatProbesJson(probesJson);
    }

    params.showTemperature = NodeParamUtils::readBoolParam(node, nodegraphparams::heatprobe::ShowTemperature, true);
    params.showName = NodeParamUtils::readBoolParam(node, nodegraphparams::heatprobe::ShowName, true);

    if (params.items.empty()) {
        // Fallback default single probe
        HeatProbeItem defaultItem{};
        defaultItem.id = 1;
        defaultItem.name = "Probe_1";
        defaultItem.worldPosition = glm::vec3{0.0f};
        defaultItem.enabled = true;
        params.items.push_back(defaultItem);
    }

    return params;
}
