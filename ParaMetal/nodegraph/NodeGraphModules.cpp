#include "NodeGraphModules.hpp"

#include "NodeBoundaryConditions.hpp"
#include "NodeCFD.hpp"
#include "NodeFluid.hpp"
#include "NodeGroup.hpp"
#include "NodeHeatModel.hpp"
#include "NodeHeatProbe.hpp"
#include "NodeHeatSolve.hpp"
#include "NodeMerge.hpp"
#include "NodeMeshPoints.hpp"
#include "NodeModel.hpp"
#include "NodePoints.hpp"
#include "NodeRemesh.hpp"
#include "NodeSerialTemperature.hpp"
#include "NodeTransform.hpp"
#include "NodeVoronoi.hpp"
#include "NodeWriteParams.hpp"

#include "NodeGraphKernels.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"

#include <memory>
#include <utility>

namespace payloadtypes {
uint8_t None = 0;
uint8_t Geometry = 0;
uint8_t Group = 0;
uint8_t Remesh = 0;
uint8_t HeatModel = 0;
uint8_t Fluid = 0;
uint8_t Heat = 0;
uint8_t Voronoi = 0;
uint8_t Points = 0;
uint8_t SerialTemperature = 0;
uint8_t CFD = 0;
uint8_t HeatProbe = 0;
uint8_t Boundary = 0;
}

struct NodeModule {
    const char* typeId;
    NodeTypeDefinition (*buildDefinition)();
    std::unique_ptr<NodeKernel> (*createKernel)();
};

static const NodeModule modules[] = {
    {nodegraphtypes::Model, &buildModelDefinition, &createModelKernel},
    {nodegraphtypes::Transform, &buildTransformDefinition, &createTransformKernel},
    {nodegraphtypes::Group, &buildGroupDefinition, &createGroupKernel},
    {nodegraphtypes::Remesh, &buildRemeshDefinition, &createRemeshKernel},
    {nodegraphtypes::HeatModel, &buildHeatModelDefinition, &createHeatModelKernel},
    {nodegraphtypes::Fluid, &buildFluidDefinition, &createFluidKernel},
    {nodegraphtypes::Voronoi, &buildVoronoiDefinition, &createVoronoiKernel},
    {nodegraphtypes::CFD, &buildCFDDefinition, &createCFDKernel},
    {nodegraphtypes::HeatSolve, &buildHeatSolveDefinition, &createHeatSolveKernel},
    {nodegraphtypes::Points, &buildPointsDefinition, &createPointsKernel},
    {nodegraphtypes::MeshPoints, &buildMeshPointsDefinition, &createMeshPointsKernel},
    {nodegraphtypes::Merge, &buildMergeDefinition, &createMergeKernel},
    {nodegraphtypes::SerialTemperature, &buildSerialTemperatureDefinition, &createSerialTemperatureKernel},
    {nodegraphtypes::HeatProbe, &buildHeatProbeDefinition, &createHeatProbeKernel},
    {nodegraphtypes::Write, &buildWriteDefinition, nullptr},
    {nodegraphtypes::BoundaryConditions, &buildBoundaryConditionsDefinition, &createBoundaryConditionsKernel},
};

static bool registerPayload(
    NodeGraphRegistry& registry,
    uint8_t& target,
    const char* name,
    NodeGraphValueType displayType) {
    target = registry.registerPayloadType(name, displayType);
    return target != 0;
}

bool registerNodePayloads(NodeGraphRegistry& registry) {
    return
        registerPayload(registry, payloadtypes::Geometry, "geometry", NodeGraphValueType::Mesh) &&
        registerPayload(registry, payloadtypes::Group, "group", NodeGraphValueType::Mesh) &&
        registerPayload(registry, payloadtypes::Remesh, "remesh", NodeGraphValueType::Remesh) &&
        registerPayload(registry, payloadtypes::HeatModel, "heat_model", NodeGraphValueType::HeatModel) &&
        registerPayload(registry, payloadtypes::Fluid, "fluid", NodeGraphValueType::Fluid) &&
        registerPayload(registry, payloadtypes::Heat, "heat", NodeGraphValueType::Simulation) &&
        registerPayload(registry, payloadtypes::Voronoi, "voronoi", NodeGraphValueType::Volume) &&
        registerPayload(registry, payloadtypes::Points, "points", NodeGraphValueType::Points) &&
        registerPayload(registry, payloadtypes::SerialTemperature, "serial_temperature", NodeGraphValueType::ScalarFloat) &&
        registerPayload(registry, payloadtypes::CFD, "cfd", NodeGraphValueType::Simulation) &&
        registerPayload(registry, payloadtypes::HeatProbe, "heat_probe", NodeGraphValueType::ScalarFloat) &&
        registerPayload(registry, payloadtypes::Boundary, "boundary", NodeGraphValueType::Boundary);
}

bool registerNodeDefinitions(NodeGraphRegistry& registry) {
    for (const NodeModule& module : modules) {
        NodeTypeDefinition definition = module.buildDefinition();
        if (definition.id != module.typeId) {
            return false;
        }
        if (!registry.registerNodeType(std::move(definition))) {
            return false;
        }
    }
    return true;
}

bool registerNodeKernels(NodeGraphKernels& kernels) {
    for (const NodeModule& module : modules) {
        if (!module.createKernel) {
            if (!kernels.registerKernellessNode(module.typeId)) {
                return false;
            }
            continue;
        }

        std::unique_ptr<NodeKernel> kernel = module.createKernel();
        if (!kernel || !kernel->typeId() ||
            NodeTypeId(kernel->typeId()) != module.typeId) {
            return false;
        }
        if (!kernels.registerKernel(std::move(kernel))) {
            return false;
        }
    }
    return true;
}
