#include "NodeHeatMaterialPresets.hpp"

#include "NodeGraphParamUtils.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/ui/widgets/NodePanelUtils.hpp"

bool tryResolveHeatPresetId(const std::string& value, HeatMaterialPresetId& outId) {
    const std::string normalized = NodePanelUtils::normalizePresetName(value);
    if (normalized == "copper") {
        outId = HeatMaterialPresetId::Copper;
        return true;
    }
    if (normalized == "iron") {
        outId = HeatMaterialPresetId::Iron;
        return true;
    }
    if (normalized == "ceramic") {
        outId = HeatMaterialPresetId::Ceramic;
        return true;
    }
    if (normalized == "custom") {
        outId = HeatMaterialPresetId::Custom;
        return true;
    }
    if (normalized == "aluminum" || normalized == "aluminium") {
        outId = HeatMaterialPresetId::Aluminum;
        return true;
    }
    return false;
}

bool tryMakeMaterialBinding(
    const std::string& modelNodeIdText,
    const std::string& presetText,
    HeatMaterialBinding& outRow) {
    uint32_t modelNodeId = 0;
    if (!NodePanelUtils::tryParseUint32Id(modelNodeIdText, modelNodeId) || modelNodeId == 0) {
        return false;
    }

    HeatMaterialPresetId presetId = HeatMaterialPresetId::Aluminum;
    if (!tryResolveHeatPresetId(presetText, presetId)) {
        return false;
    }

    outRow.modelNodeId = modelNodeId;
    outRow.presetId = presetId;
    return true;
}

std::vector<HeatMaterialBinding> readMaterialBindings(const NodeGraphParamValue& value) {
    std::vector<HeatMaterialBinding> rows;
    if (value.type != NodeGraphParamType::Array) {
        return rows;
    }

    for (const NodeGraphParamValue& elementValue : value.arrayValues) {
        if (elementValue.type != NodeGraphParamType::Struct) {
            continue;
        }

        const NodeGraphParamValue* receiverModelField = findParamFieldValue(elementValue, "modelNodeId");
        const NodeGraphParamValue* presetField = findParamFieldValue(elementValue, "preset");
        if (!receiverModelField || !presetField) {
            continue;
        }

        int64_t modelNodeId = 0;
        if (!tryGetParamInt(*receiverModelField, modelNodeId) || modelNodeId <= 0) {
            continue;
        }
        std::string presetName;
        if (!tryGetParamEnum(*presetField, presetName)) {
            continue;
        }

        HeatMaterialBinding row{};
        if (!tryMakeMaterialBinding(std::to_string(modelNodeId), presetName, row)) {
            continue;
        }
        rows.push_back(std::move(row));
    }

    return rows;
}

bool writeMaterialBindings(NodeGraphParamValue& value, const std::vector<HeatMaterialBinding>& rows) {
    if (value.type != NodeGraphParamType::Array) {
        return false;
    }

    value.arrayValues.clear();
    value.arrayValues.reserve(rows.size());
    for (const HeatMaterialBinding& row : rows) {
        if (!addArrayElement(
                value,
                makeStructParamValue({
                    makeParamFieldValue(
                        "modelNodeId",
                        makeIntParamValue(static_cast<int64_t>(row.modelNodeId))),
                    makeParamFieldValue(
                        "preset",
                        makeEnumParamValue(heatMaterialPresetName(row.presetId))),
                }))) {
            return false;
        }
    }

    return true;
}
