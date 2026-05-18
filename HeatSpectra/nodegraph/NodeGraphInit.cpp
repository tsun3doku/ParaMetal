#include "NodeGraphInit.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphTypes.hpp"
#include "NodeGraphParamUtils.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <cstdint>

namespace payloadtypes {
    uint8_t None = 0;
    uint8_t Geometry = 0;
    uint8_t Remesh = 0;
    uint8_t HeatModel = 0;
    uint8_t Heat = 0;
    uint8_t Voronoi = 0;
    uint8_t Contact = 0;
}

namespace {

NodeSocketSignature makeInputSocket(
    const char* name,
    NodeGraphValueType valueType,
    bool variadic = false) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Input;
    signature.valueType = valueType;
    signature.variadic = variadic;
    return signature;
}

NodeSocketSignature makeOutputSocket(
    const char* name,
    NodeGraphValueType valueType,
    uint8_t producedPayloadType) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Output;
    signature.valueType = valueType;
    signature.contract.producedPayloadType = producedPayloadType;
    return signature;
}

NodeTypeDefinition buildModelNode() {
    return {
        nodegraphtypes::Model,
        "Model",
        NodeGraphNodeCategory::Model,
        {
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Geometry),
        },
        {
            {nodegraphparams::model::Path, "Model Path", NodeGraphParamType::String, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildTransformNode() {
    return {
        nodegraphtypes::Transform,
        "Transform",
        NodeGraphNodeCategory::Meshing,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Geometry),
        },
        {
            {nodegraphparams::transform::TranslateX, "Translate X", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::TranslateY, "Translate Y", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::TranslateZ, "Translate Z", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateXDegrees, "Rotate X", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateYDegrees, "Rotate Y", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateZDegrees, "Rotate Z", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleX, "Scale X", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleY, "Scale Y", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleZ, "Scale Z", NodeGraphParamType::Float, 1.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildGroupNode() {
    return {
        nodegraphtypes::Group,
        "Group",
        NodeGraphNodeCategory::Meshing,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Geometry),
        },
        {
            {nodegraphparams::group::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::group::SourceType, "Source Type", NodeGraphParamType::Int, 0.0, nodegraphparams::group::sourcetype::Vertex, false, "", false},
            {nodegraphparams::group::SourceName, "Source Name", NodeGraphParamType::String, 0.0, 0, false, "", false},
            {nodegraphparams::group::TargetName, "Target Group Name", NodeGraphParamType::String, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildRemeshNode() {
    return {
        nodegraphtypes::Remesh,
        "Remesh",
        NodeGraphNodeCategory::Meshing,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Remesh),
        },
        {
            {nodegraphparams::remesh::Iterations, "Iterations", NodeGraphParamType::Int, 0.0, 1, false, "", false},
            {nodegraphparams::remesh::MinAngleDegrees, "Min Angle", NodeGraphParamType::Float, 20.0, 0, false, "", false},
            {nodegraphparams::remesh::MaxEdgeLength, "Max Edge Length", NodeGraphParamType::Float, 0.1, 0, false, "", false},
            {nodegraphparams::remesh::StepSize, "Step Size", NodeGraphParamType::Float, 0.25, 0, false, "", false},
            {nodegraphparams::remesh::RunRequested, "Run Requested", NodeGraphParamType::Bool, 0.0, 0, false, "", true},
            {nodegraphparams::remesh::ShowRemeshOverlay, "Show Remesh Overlay", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::remesh::ShowFaceNormals, "Show Face Normals", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::remesh::ShowVertexNormals, "Show Vertex Normals", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::remesh::NormalLength, "Normal Length", NodeGraphParamType::Float, 0.05, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildHeatModelNode() {
    return {
        nodegraphtypes::HeatModel,
        "Heat Model",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("HeatModel", NodeGraphValueType::Mesh, payloadtypes::HeatModel),
        },
        {
            {nodegraphparams::heatmodel::Density, "Density", NodeGraphParamType::Float, HeatSimDefaults::density, 0, false, "", false},
            {nodegraphparams::heatmodel::SpecificHeat, "Specific Heat", NodeGraphParamType::Float, HeatSimDefaults::specificHeat, 0, false, "", false},
            {nodegraphparams::heatmodel::Conductivity, "Conductivity", NodeGraphParamType::Float, HeatSimDefaults::conductivity, 0, false, "", false},
            {nodegraphparams::heatmodel::InitialTemperature, "Initial Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperature, 0, false, "", false},
            makeEnumParamDefinition(
                nodegraphparams::heatmodel::BoundaryCondition,
                "Boundary Condition",
                "None",
                {"None", "Fixed Temperature", "Fixed Power"}),
            {nodegraphparams::heatmodel::FixedTemperatureValue, "Fixed Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperature, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildContactNode() {
    return {
        nodegraphtypes::Contact,
        "Contact",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Emitter", NodeGraphValueType::Mesh),
            makeInputSocket("Receiver", NodeGraphValueType::Mesh),
            makeOutputSocket("Field", NodeGraphValueType::Field, payloadtypes::Contact),
        },
        {
            {nodegraphparams::contact::MinNormalDot, "Min Normal Dot", NodeGraphParamType::Float, HeatSimDefaults::minNormalDot, 0, false, "", false},
            {nodegraphparams::contact::ContactRadius, "Contact Radius", NodeGraphParamType::Float, HeatSimDefaults::contactRadius, 0, false, "", false},
            {nodegraphparams::contact::ShowContactLines, "Show Contact Lines", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildVoronoiNode() {
    return {
        nodegraphtypes::Voronoi,
        "Voronoi",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh, true),
            makeOutputSocket("Volume", NodeGraphValueType::Volume, payloadtypes::Voronoi),
        },
        {
            {nodegraphparams::voronoi::CellSize, "Cell Size", NodeGraphParamType::Float, 0.005, 0, false, "", false},
            {nodegraphparams::voronoi::VoxelResolution, "Voxel Resolution", NodeGraphParamType::Int, 0.0, 128, false, "", false},
            {nodegraphparams::voronoi::ShowVoronoi, "Show Voronoi", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::voronoi::ShowPoints, "Show Points", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildHeatSolveNode() {
    NodeGraphParamDefinition materialBindingDefinition = makeStructParamDefinition(
        0,
        "Material Binding",
        {
            makeParamField("receiverModelNodeId", makeIntParamDefinition(0, "Receiver Model Node ID")),
            makeParamField(
                "preset",
                makeEnumParamDefinition(
                    0,
                    "Preset",
                    "Aluminum",
                    {"Aluminum", "Copper", "Custom", "Iron", "Ceramic"})),
        });

    return {
        nodegraphtypes::HeatSolve,
        "Heat Solve",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Volume", NodeGraphValueType::Volume, true),
            makeInputSocket("Field", NodeGraphValueType::Field, true),
            makeOutputSocket("Heat", NodeGraphValueType::None, payloadtypes::Heat),
        },
        {
            {nodegraphparams::heatsolve::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::Paused, "Paused", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ResetRequested, "Reset Requested", NodeGraphParamType::Bool, 0.0, 0, false, "", true},
            makeArrayParamDefinition(nodegraphparams::heatsolve::MaterialBindings, "Material Bindings", std::move(materialBindingDefinition)),
            {nodegraphparams::heatsolve::CellSize, "Cell Size", NodeGraphParamType::Float, 0.005, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowHeatOverlay, "Show Heat Overlay", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ContactThermalConductance, "Contact Thermal Conductance", NodeGraphParamType::Float, HeatSimDefaults::contactThermalConductance, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowFluxVectors, "Show Flux Vectors", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::FluxVectorScale, "Flux Vector Scale", NodeGraphParamType::Float, 1.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildCustomNode() {
    return {
        nodegraphtypes::Custom,
        "Custom",
        NodeGraphNodeCategory::Custom,
        {
            makeInputSocket("In", NodeGraphValueType::None),
            makeOutputSocket("Out", NodeGraphValueType::None, 0),
        },
        {},
    };
}

} // anonymous namespace

void initNodeGraph(NodeGraphRegistry& registry) {
    payloadtypes::Geometry   = registry.registerPayloadType("geometry", NodeGraphValueType::Mesh);
    payloadtypes::Remesh     = registry.registerPayloadType("remesh",   NodeGraphValueType::Mesh);
    payloadtypes::HeatModel  = registry.registerPayloadType("heat_model", NodeGraphValueType::Mesh);
    payloadtypes::Heat       = registry.registerPayloadType("heat",      NodeGraphValueType::None);
    payloadtypes::Voronoi    = registry.registerPayloadType("voronoi",   NodeGraphValueType::Volume);
    payloadtypes::Contact    = registry.registerPayloadType("contact",   NodeGraphValueType::Field);

    registry.registerNodeType(buildModelNode());
    registry.registerNodeType(buildTransformNode());
    registry.registerNodeType(buildGroupNode());
    registry.registerNodeType(buildRemeshNode());
    registry.registerNodeType(buildHeatModelNode());
    registry.registerNodeType(buildContactNode());
    registry.registerNodeType(buildVoronoiNode());
    registry.registerNodeType(buildHeatSolveNode());
    registry.registerNodeType(buildCustomNode());
}
