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

bool tryMakeMaterialBindingRow(
    const std::string& modelNodeIdText,
    const std::string& presetText,
    HeatMaterialBindingRow& outRow) {
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

std::vector<HeatMaterialBindingRow> readMaterialBindingRows(const NodeGraphParamValue& value) {
    std::vector<HeatMaterialBindingRow> rows;
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

        HeatMaterialBindingRow row{};
        if (!tryMakeMaterialBindingRow(std::to_string(modelNodeId), presetName, row)) {
            continue;
        }
        rows.push_back(std::move(row));
    }

    return rows;
}

bool writeMaterialBindingRows(NodeGraphParamValue& value, const std::vector<HeatMaterialBindingRow>& rows) {
    if (value.type != NodeGraphParamType::Array) {
        return false;
    }

    value.arrayValues.clear();
    value.arrayValues.reserve(rows.size());
    for (const HeatMaterialBindingRow& row : rows) {
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

HeatMaterialBinding makeHeatMaterialBinding(const HeatMaterialBindingRow& row) {
    HeatMaterialBinding binding{};
    binding.modelNodeId = row.modelNodeId;
    binding.presetId = row.presetId;
    return binding;
}

std::vector<HeatMaterialBinding> makeHeatMaterialBindings(const std::vector<HeatMaterialBindingRow>& rows) {
    std::vector<HeatMaterialBinding> bindings;
    bindings.reserve(rows.size());
    for (const HeatMaterialBindingRow& row : rows) {
        HeatMaterialBinding binding = makeHeatMaterialBinding(row);
        bindings.push_back(std::move(binding));
    }
    return bindings;
}
