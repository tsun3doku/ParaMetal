#include "NodeFluid.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeFluidParams.hpp"
#include "NodePayloadRegistry.hpp"

static NodeTypeDefinition buildFluidNode() {
    return {
        nodegraphtypes::Fluid,
        "Fluid",
        NodeGraphNodeCategory::System,
        {
            makeOutputSocket("Fluid", NodeGraphValueType::Fluid, payloadtypes::Fluid),
        },
        {
            {nodegraphparams::fluid::Density, "Density", NodeGraphParamType::Float, 1.225, 0, false, "", false},
            {nodegraphparams::fluid::SpecificHeat, "Specific Heat", NodeGraphParamType::Float, 1005.0, 0, false, "", false},
            {nodegraphparams::fluid::Conductivity, "Conductivity", NodeGraphParamType::Float, 0.026, 0, false, "", false},
            {nodegraphparams::fluid::InitialTemperatureC, "Initial Temperature (C)", NodeGraphParamType::Float, 20.0, 0, false, "", false},
            {nodegraphparams::fluid::KinematicViscosity, "Kinematic Viscosity (m2/s)", NodeGraphParamType::Float, 1.516e-5, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildFluidDefinition() {
    return buildFluidNode();
}

std::unique_ptr<NodeKernel> createFluidKernel() {
    return std::make_unique<NodeFluid>();
}

const char* NodeFluid::typeId() const {
    return nodegraphtypes::Fluid;
}

void NodeFluid::execute(NodeKernelEval& eval) const {
    const FluidNodeParams params = readFluidNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Fluid) {
            continue;
        }

        FluidData payload{};
        payload.density = static_cast<float>(params.density);
        payload.specificHeat = static_cast<float>(params.specificHeat);
        payload.conductivity = static_cast<float>(params.conductivity);
        payload.initialTemperatureC = static_cast<float>(params.initialTemperatureC);
        payload.kinematicViscosity = static_cast<float>(params.kinematicViscosity);

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeFluid::computeOutputHashes(const NodeKernelHash& hash) const {
    const FluidNodeParams params = readFluidNodeParams(hash.node);
    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineString(propertiesHash, nodegraphtypes::Fluid);
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.density));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.specificHeat));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.conductivity));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.initialTemperatureC));
    HashBuilder::combineFloat(propertiesHash, static_cast<float>(params.kinematicViscosity));

    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, propertiesHash);

    HashValues values{};
    values.full = simulationHash;
    values.properties = propertiesHash;
    values.simulation = simulationHash;
    return values;
}
