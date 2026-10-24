#include "NodeFluidParams.hpp"
#include "NodeGraphEditor.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodeParamUtils.hpp"

FluidNodeParams readFluidNodeParams(const NodeGraphNode& node) {
    FluidNodeParams params{};
    params.density = NodeParamUtils::readFloatParam(node, nodegraphparams::fluid::Density, 1.225);
    params.specificHeat = NodeParamUtils::readFloatParam(node, nodegraphparams::fluid::SpecificHeat, 1005.0);
    params.conductivity = NodeParamUtils::readFloatParam(node, nodegraphparams::fluid::Conductivity, 0.026);
    params.initialTemperatureC = NodeParamUtils::readFloatParam(node, nodegraphparams::fluid::InitialTemperatureC, 20.0);
    params.kinematicViscosity = NodeParamUtils::readFloatParam(node, nodegraphparams::fluid::KinematicViscosity, 1.516e-5);
    return params;
}

bool writeFluidNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const FluidNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::fluid::Density, NodeGraphParamType::Float, params.density}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::fluid::SpecificHeat, NodeGraphParamType::Float, params.specificHeat}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::fluid::Conductivity, NodeGraphParamType::Float, params.conductivity}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::fluid::InitialTemperatureC, NodeGraphParamType::Float, params.initialTemperatureC}) &&
           editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::fluid::KinematicViscosity, NodeGraphParamType::Float, params.kinematicViscosity});
}
