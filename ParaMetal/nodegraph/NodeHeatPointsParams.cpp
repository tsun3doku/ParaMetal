#include "NodeHeatPointsParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

HeatPointsNodeParams readHeatPointsNodeParams(const NodeGraphNode& node) {
    HeatPointsNodeParams params{};
    tryGetNodeParamFloat(node, nodegraphparams::heatpoints::InitialTemperature, params.initialTemperature);
    tryGetNodeParamFloat(node, nodegraphparams::heatpoints::FixedTemperature, params.fixedTemperature);

    int64_t bcValue = 0;
    tryGetNodeParamInt(node, nodegraphparams::heatpoints::BoundaryCondition, bcValue);
    if (bcValue >= 0 && bcValue <= 2) {
        params.boundaryCondition = static_cast<HeatBoundaryCondition>(bcValue);
    }

    const NodeGraphParamValue* rowsParam = findNodeParamValue(node, nodegraphparams::heatpoints::Points);
    if (!rowsParam || rowsParam->type != NodeGraphParamType::Array) {
        return params;
    }

    params.rows.reserve(rowsParam->arrayValues.size());
    for (const NodeGraphParamValue& rowValue : rowsParam->arrayValues) {
        if (rowValue.type != NodeGraphParamType::Struct) {
            continue;
        }

        HeatPointNodeRow row{};
        double fixedTemp = row.fixedTemperature;
        int64_t bcType = 0;
        tryGetFieldInt(rowValue, "bcType", bcType);
        tryGetFieldFloat(rowValue, "fixedTemp", fixedTemp);
        row.boundaryCondition = bcType > 0 ? static_cast<uint32_t>(bcType) : 0u;
        row.fixedTemperature = static_cast<float>(fixedTemp);
        params.rows.push_back(row);
    }

    return params;
}

bool writeHeatPointsNodeParams(
    NodeGraphEditor& editor,
    NodeGraphNodeId nodeId,
    const HeatPointsNodeParams& params) {
    bool ok = true;
    ok = editor.setNodeParameter(
            nodeId,
            NodeGraphParamValue{
                nodegraphparams::heatpoints::InitialTemperature,
                NodeGraphParamType::Float,
                params.initialTemperature}) && ok;
    ok = editor.setNodeParameter(
            nodeId,
            NodeGraphParamValue{
                nodegraphparams::heatpoints::FixedTemperature,
                NodeGraphParamType::Float,
                params.fixedTemperature}) && ok;
    ok = editor.setNodeParameter(
            nodeId,
            NodeGraphParamValue{
                nodegraphparams::heatpoints::BoundaryCondition,
                NodeGraphParamType::Int,
                0.0,
                static_cast<int64_t>(params.boundaryCondition)}) && ok;

    std::vector<NodeGraphParamValue> rowValues;
    rowValues.reserve(params.rows.size());
    for (const HeatPointNodeRow& row : params.rows) {
        rowValues.push_back(makeStructParamValue({
            makeParamFieldValue("bcType", makeIntParamValue(static_cast<int64_t>(row.boundaryCondition))),
            makeParamFieldValue("fixedTemp", makeFloatParamValue(row.fixedTemperature)),
        }));
    }

    ok = editor.setNodeParameter(nodeId, makeArrayParamValue(nodegraphparams::heatpoints::Points, std::move(rowValues))) && ok;

    return ok;
}
