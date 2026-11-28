#include "NodeHeatModel.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeHeatModelParams.hpp"
#include "NodePayloadRegistry.hpp"
#include "heat/HeatSystemPresets.hpp"

static NodeTypeDefinition buildHeatModelNode() {
    return {
        nodegraphtypes::HeatModel,
        "Heat Model",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Remesh", NodeGraphValueType::Remesh),
            makeOutputSocket("HeatModel", NodeGraphValueType::HeatModel, payloadtypes::HeatModel),
        },
        {
            makeEnumParamDefinition(
                nodegraphparams::heatmodel::MaterialPreset,
                "Material Preset",
                "Custom",
                {"Aluminum", "Copper", "Custom", "Iron", "Ceramic"}),
            {nodegraphparams::heatmodel::Density, "Density", NodeGraphParamType::Float, HeatSimDefaults::density, 0, false, "", false},
            {nodegraphparams::heatmodel::SpecificHeat, "Specific Heat", NodeGraphParamType::Float, HeatSimDefaults::specificHeat, 0, false, "", false},
            {nodegraphparams::heatmodel::Conductivity, "Conductivity", NodeGraphParamType::Float, HeatSimDefaults::conductivity, 0, false, "", false},
            {nodegraphparams::heatmodel::InitialTemperatureC, "Initial Temperature (C)", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatmodel::VolumetricPowerDensity, "Volumetric Power Density (W/m³)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatmodel::PrescribedTemperatureEnabled, "Fixed Temperature", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatmodel::PrescribedTemperatureC, "Temperature (C)", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildHeatModelDefinition() {
    return buildHeatModelNode();
}

std::unique_ptr<NodeKernel> createHeatModelKernel() {
    return std::make_unique<NodeHeatModel>();
}

const char* NodeHeatModel::typeId() const {
    return nodegraphtypes::HeatModel;
}

void NodeHeatModel::execute(NodeKernelEval& eval) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(eval.node);
    const std::size_t meshSocketIndex = inputIndexOf(eval.node, NodeGraphValueType::Remesh);
    const NodeDataBlock* inputMeshValue =
        (meshSocketIndex < eval.inputs.size() && !eval.inputs[meshSocketIndex].empty())
            ? eval.inputs[meshSocketIndex].front() : nullptr;
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    NodeDataHandle meshHandle{};
    const bool hasValidInput = payloadRegistry && inputMeshValue && inputMeshValue->payloadHandle.key != 0;
    if (hasValidInput) {
        payloadRegistry->resolveRemesh(inputMeshValue->payloadHandle, &meshHandle);
    }

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::HeatModel || meshHandle.key == 0) {
            continue;
        }

        HeatModelData payload{};
        payload.meshHandle = meshHandle;
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperatureC = static_cast<float>(params.initialTemperatureC);
        payload.volumetricHeatSource.powerDensity = static_cast<float>(params.volumetricPowerDensity);
        payload.prescribedTemperatureEnabled = params.prescribedTemperatureEnabled;
        payload.prescribedTemperatureC = static_cast<float>(params.prescribedTemperatureC);
        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeHeatModel::computeOutputHashes(const NodeKernelHash& hash) const {
    const HeatModelNodeParams params = readHeatModelNodeParams(hash.node);
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::HeatModel);
    HashNodeCache::combineSocket(geometryHash, hash, NodeGraphValueType::Remesh, HashDomain::Geometry);

    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.initialTemperatureC));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.volumetricPowerDensity));
    HashBuilder::combine(propertiesHash, params.prescribedTemperatureEnabled ? 1u : 0u);
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.prescribedTemperatureC));

    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, geometryHash);
    HashBuilder::combine(simulationHash, propertiesHash);

    HashValues values{};
    values.full = simulationHash;
    values.geometry = geometryHash;
    values.properties = propertiesHash;
    values.simulation = simulationHash;
    return values;
}
