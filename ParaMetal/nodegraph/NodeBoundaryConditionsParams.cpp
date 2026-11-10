#include "NodeBoundaryConditionsParams.hpp"

#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

BoundaryConditionsNodeParams readBoundaryConditionsNodeParams(const NodeGraphNode& node) {
    BoundaryConditionsNodeParams params{};
    for (uint32_t face = 0; face < 6; ++face) {
        BoundaryData::Face& values = params.faces[face];
        values.flowType = static_cast<BoundaryData::FlowType>(
            NodeParamUtils::readEnumParam(node, nodegraphparams::boundaryconditions::flowTypeParam(face), 0));
        values.velocity.x = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::velocityXParam(face), 0.0));
        values.velocity.y = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::velocityYParam(face), 0.0));
        values.velocity.z = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::velocityZParam(face), 0.0));
        values.thermalType = static_cast<BoundaryData::ThermalType>(
            NodeParamUtils::readEnumParam(node, nodegraphparams::boundaryconditions::thermalTypeParam(face), 0));
        values.temperatureC = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::temperatureParam(face), 5.0));
        values.heatFlux = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::heatFluxParam(face), 0.0));
        values.htc = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::htcParam(face), 0.0));
        values.inflowTemperatureC = static_cast<float>(
            NodeParamUtils::readFloatParam(node, nodegraphparams::boundaryconditions::inflowTemperatureParam(face), 20.0));
    }
    return params;
}
