#include "NodeBoundaryConditions.hpp"
#include "NodeBoundaryConditionsParams.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"

#include "hash/HashBuilder.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

#include <cstdint>
#include <string>
#include <utility>

static NodeTypeDefinition buildBoundaryConditionsNode() {
    const char* const faceNames[6] = { "-X", "+X", "-Y", "+Y", "-Z", "+Z" };
    const char* const defaultFlow[6] = {
        "Velocity Inlet", "Pressure Outlet", "No-Slip Wall", "No-Slip Wall", "No-Slip Wall", "No-Slip Wall"
    };
    const double defaultVelocityX[6] = { 0.1, 0.0, 0.0, 0.0, 0.0, 0.0 };

    std::vector<NodeGraphParamDefinition> parameters;
    parameters.reserve(54);
    for (uint32_t face = 0; face < 6; ++face) {
        const std::string prefix = std::string(faceNames[face]) + " ";
        parameters.push_back(makeEnumParamDefinition(
            nodegraphparams::boundaryconditions::flowTypeParam(face),
            (prefix + "Flow").c_str(), defaultFlow[face],
            {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}));
        parameters.push_back({nodegraphparams::boundaryconditions::velocityXParam(face), prefix + "Velocity X (m/s)", NodeGraphParamType::Float, defaultVelocityX[face], 0, false, "", false});
        parameters.push_back({nodegraphparams::boundaryconditions::velocityYParam(face), prefix + "Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false});
        parameters.push_back({nodegraphparams::boundaryconditions::velocityZParam(face), prefix + "Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false});
        parameters.push_back(makeEnumParamDefinition(
            nodegraphparams::boundaryconditions::thermalTypeParam(face),
            (prefix + "Thermal").c_str(), "Adiabatic",
            {"Adiabatic", "Dirichlet", "Neumann", "Robin"}));
        parameters.push_back({nodegraphparams::boundaryconditions::temperatureParam(face), prefix + "Temperature (C)", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false});
        parameters.push_back({nodegraphparams::boundaryconditions::heatFluxParam(face), prefix + "Heat Flux (W/m2)", NodeGraphParamType::Float, 0.0, 0, false, "", false});
        parameters.push_back({nodegraphparams::boundaryconditions::htcParam(face), prefix + "HTC (W/(m2 K))", NodeGraphParamType::Float, 0.0, 0, false, "", false});
        parameters.push_back({nodegraphparams::boundaryconditions::inflowTemperatureParam(face), prefix + "Inflow Temperature (C)", NodeGraphParamType::Float, 20.0, 0, false, "", false});
    }

    return {
        nodegraphtypes::BoundaryConditions,
        "Boundary Conditions",
        NodeGraphNodeCategory::System,
        {
            makeOutputSocket("Boundary", NodeGraphValueType::Boundary, payloadtypes::Boundary),
        },
        std::move(parameters),
    };
}

NodeTypeDefinition buildBoundaryConditionsDefinition() {
    return buildBoundaryConditionsNode();
}

std::unique_ptr<NodeKernel> createBoundaryConditionsKernel() {
    return std::make_unique<NodeBoundaryConditions>();
}

const char* NodeBoundaryConditions::typeId() const {
    return nodegraphtypes::BoundaryConditions;
}

void NodeBoundaryConditions::execute(NodeKernelEval& eval) const {
    const BoundaryConditionsNodeParams params = readBoundaryConditionsNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0;
         outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size();
         ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Boundary) {
            continue;
        }

        BoundaryData payload{};
        payload.faces = params.faces;

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeBoundaryConditions::computeOutputHashes(const NodeKernelHash& hash) const {
    const BoundaryConditionsNodeParams params = readBoundaryConditionsNodeParams(hash.node);

    uint64_t boundaryHash = HashBuilder::start();
    HashBuilder::combineString(boundaryHash, nodegraphtypes::BoundaryConditions);
    for (const BoundaryData::Face& face : params.faces) {
        HashBuilder::combine(boundaryHash, static_cast<uint64_t>(face.flowType));
        HashBuilder::combinePod(boundaryHash, face.velocity);
        HashBuilder::combine(boundaryHash, static_cast<uint64_t>(face.thermalType));
        HashBuilder::combineFloat(boundaryHash, face.temperatureC);
        HashBuilder::combineFloat(boundaryHash, face.heatFlux);
        HashBuilder::combineFloat(boundaryHash, face.htc);
        HashBuilder::combineFloat(boundaryHash, face.inflowTemperatureC);
    }

    HashValues values{};
    values.full = boundaryHash;
    values.properties = boundaryHash;
    values.simulation = boundaryHash;
    return values;
}
