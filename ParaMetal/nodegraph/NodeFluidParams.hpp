#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/FluidData.hpp"

class NodeGraphEditor;
struct NodeGraphNode;

struct FluidNodeParams {
    double density = 1.225;
    double specificHeat = 1005.0;
    double conductivity = 0.026;
    double kinematicViscosity = 1.516e-5;
    double initialTemperatureC = 20.0;
};

using HeatFluidNodeParams = FluidNodeParams;

FluidNodeParams readFluidNodeParams(const NodeGraphNode& node);
bool writeFluidNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const FluidNodeParams& params);

inline HeatFluidNodeParams readHeatFluidNodeParams(const NodeGraphNode& node) {
    return readFluidNodeParams(node);
}

inline bool writeHeatFluidNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const HeatFluidNodeParams& params) {
    return writeFluidNodeParams(editor, nodeId, params);
}

namespace nodegraphparams::heatfluid {
constexpr uint32_t Density = 1;
constexpr uint32_t SpecificHeat = 2;
constexpr uint32_t Conductivity = 3;
constexpr uint32_t InitialTemperatureC = 4;
constexpr uint32_t KinematicViscosity = 9;
}

namespace nodegraphparams {
namespace fluid = heatfluid;
}
