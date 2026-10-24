#include "NodeGraphInit.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphTypes.hpp"
#include "NodeGraphParamUtils.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "voronoi/VoronoiConfig.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

#include <cstdint>
#include <initializer_list>

namespace payloadtypes {
    uint8_t None = 0;
    uint8_t Geometry = 0;
    uint8_t Remesh = 0;
    uint8_t HeatModel = 0;
    uint8_t Fluid = 0;
    uint8_t Heat = 0;
    uint8_t Voronoi = 0;
    uint8_t Points = 0;
    uint8_t SerialTemperature = 0;
    uint8_t CFD = 0;
}

static NodeSocketSignature makeInputSocket(
    const char* name,
    NodeGraphValueType valueType,
    bool variadic = false,
    bool required = true) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Input;
    signature.valueType = valueType;
    signature.variadic = variadic;
    signature.required = required;
    return signature;
}

static NodeSocketSignature makeOutputSocket(
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

static NodeSocketSignature makePolymorphicInputSocket(
    const char* name,
    std::initializer_list<NodeGraphValueType> acceptedValueTypes,
    bool variadic = false,
    bool required = true) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Input;
    signature.valueType = NodeGraphValueType::None;
    signature.acceptedValueTypes = acceptedValueTypes;
    signature.variadic = variadic;
    signature.required = required;
    return signature;
}

static NodeSocketSignature makePolymorphicOutputSocket(
    const char* name,
    std::initializer_list<NodeGraphValueType> acceptedValueTypes,
    uint8_t producedPayloadType) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Output;
    signature.valueType = NodeGraphValueType::None;
    signature.acceptedValueTypes = acceptedValueTypes;
    signature.contract.producedPayloadType = producedPayloadType;
    return signature;
}

static NodeTypeDefinition buildModelNode() {
    return {
        nodegraphtypes::Model,
        "Model",
        NodeGraphNodeCategory::Geometry,
        {
            makeOutputSocket("Mesh", NodeGraphValueType::Mesh, payloadtypes::Geometry),
        },
        {
            {nodegraphparams::model::Path, "Model Path", NodeGraphParamType::String, 0.0, 0, false, "", false},
            makeEnumParamDefinition(nodegraphparams::model::SourceUnit, "Import Units", "Meters", {"Meters", "Centimeters", "Millimeters"}),
        },
    };
}

static NodeTypeDefinition buildTransformNode() {
    return {
        nodegraphtypes::Transform,
        "Transform",
        NodeGraphNodeCategory::Meshing,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, false, false),
            makePolymorphicOutputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, payloadtypes::None),
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

static NodeTypeDefinition buildGroupNode() {
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

static NodeTypeDefinition buildRemeshNode() {
    return {
        nodegraphtypes::Remesh,
        "Remesh",
        NodeGraphNodeCategory::Meshing,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("Remesh", NodeGraphValueType::Remesh, payloadtypes::Remesh),
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

static NodeTypeDefinition buildSerialTemperatureNode() {
    return {
        nodegraphtypes::SerialTemperature,
        "Serial Temperature",
        NodeGraphNodeCategory::System,
        {
            makeOutputSocket(
                "Temperature", NodeGraphValueType::ScalarFloat, payloadtypes::SerialTemperature),
        },
        {
            {nodegraphparams::serialtemperature::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::serialtemperature::Port, "Port", NodeGraphParamType::String, 0.0, 0, false, "", false},
            {nodegraphparams::serialtemperature::BaudRate, "Baud Rate", NodeGraphParamType::Int, 0.0, 115200, false, "", false},
        },
    };
}



static NodeTypeDefinition buildVoronoiNode() {
    return {
        nodegraphtypes::Voronoi,
        "Voronoi",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Remeshes", NodeGraphValueType::Remesh, true, false),
            makeInputSocket("Points", NodeGraphValueType::Points, false, true),
            makeOutputSocket("Volume", NodeGraphValueType::Volume, payloadtypes::Voronoi),
        },
        {
            {nodegraphparams::voronoi::CellSize, "Cell Size", NodeGraphParamType::Float, 0.005, 0, false, "", false},
            {nodegraphparams::voronoi::VoxelResolution, "Voxel Resolution", NodeGraphParamType::Int, 0.0, 128, false, "", false},
            {nodegraphparams::voronoi::VolumePadding, "Volume Grid Padding", NodeGraphParamType::Float, 0.01, 0, false, "", false},
            {nodegraphparams::voronoi::ContactRange, "Contact Range", NodeGraphParamType::Float, voronoi::DefaultContactRange, 0, false, "", false},
            {nodegraphparams::voronoi::ShowVoronoi, "Show Voronoi", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::voronoi::ShowPoints, "Show Points", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::voronoi::ShowContactManifold, "Show Contact Manifold", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
        },
    };
}

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

static NodeTypeDefinition buildHeatSolveNode() {
    return {
        nodegraphtypes::HeatSolve,
        "Ember",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Volume", NodeGraphValueType::Volume),
            makeInputSocket("HeatModel", NodeGraphValueType::HeatModel, true, false),
            makeInputSocket("Fluid", NodeGraphValueType::Fluid, false, false),
            makeInputSocket("CFD", NodeGraphValueType::Simulation, false, false),
            makeOutputSocket("Heat", NodeGraphValueType::Simulation, payloadtypes::Heat),
        },
        {
            {nodegraphparams::heatsolve::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::Paused, "Paused", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ResetRequested, "Reset Requested", NodeGraphParamType::Int, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowHeatOverlay, "Show Heat Overlay", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::ShowFluxVectors, "Show Flux Vectors", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowHeatPalette, "Show Heat Palette", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::FluxVectorScale, "Flux Vector Scale", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::heatsolve::RewindFrame, "Rewind Frame", NodeGraphParamType::Int, 0.0, static_cast<int64_t>(heat::NoRewindFrame), false, "", false},
            {nodegraphparams::heatsolve::SimulationDuration, "Simulation Duration", NodeGraphParamType::Float, 10.0, 0, false, "", false},
            {nodegraphparams::heatsolve::ShowDomainBox, "Show Domain Box", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::ShowFluidVolume, "Show Fluid Volume", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::heatsolve::FluidVolumeOpacity, "Fluid Volume Opacity", NodeGraphParamType::Float, 0.25, 0, false, "", false},
            {nodegraphparams::heatsolve::SolverMaxIterations, "Max PCG Iterations (0=default)", NodeGraphParamType::Int, 100.0, 100, false, "", false},
            {nodegraphparams::heatsolve::SolverConvergenceTolerance, "PCG Tolerance (0=default)", NodeGraphParamType::Float, 3e-3, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(0),
                "-X Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(0), "-X Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(0), "-X Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(0), "-X HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(1),
                "+X Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(1), "+X Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(1), "+X Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(1), "+X HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(2),
                "-Y Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(2), "-Y Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(2), "-Y Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(2), "-Y HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(3),
                "+Y Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(3), "+Y Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(3), "+Y Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(3), "+Y HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(4),
                "-Z Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(4), "-Z Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(4), "-Z Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(4), "-Z HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::heatsolve::domainBoundaryTypeParam(5),
                "+Z Boundary Type", "Adiabatic",
                {"Adiabatic", "Dirichlet", "Neumann", "Robin"}),
            {nodegraphparams::heatsolve::domainBoundaryTempParam(5), "+Z Temperature", NodeGraphParamType::Float, HeatSimDefaults::ambientTemperatureC, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryFluxParam(5), "+Z Heat Flux", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::heatsolve::domainBoundaryHtcParam(5), "+Z HTC", NodeGraphParamType::Float, 0.0, 0, false, "", false},
        },
    };
}

static NodeTypeDefinition buildCFDNode() {
    return {
        nodegraphtypes::CFD,
        "CFD",
        NodeGraphNodeCategory::System,
        {
            makeInputSocket("Volume", NodeGraphValueType::Volume, false, true),
            makeInputSocket("Fluid", NodeGraphValueType::Fluid, false, false),
            makeOutputSocket("CFD", NodeGraphValueType::Simulation, payloadtypes::CFD),
        },
        {
            {nodegraphparams::cfd::Enabled, "Enabled", NodeGraphParamType::Bool, 0.0, 0, true, "", false},
            {nodegraphparams::cfd::TargetCourant, "Target Courant", NodeGraphParamType::Float, 0.5, 0, false, "", false},
            {nodegraphparams::cfd::TimeStep, "Time Step (s)", NodeGraphParamType::Float, 1.0 / 60.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityX, "Initial Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityY, "Initial Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::VelocityZ, "Initial Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::PressureIterations, "Pressure Iterations", NodeGraphParamType::Int, 200.0, 200, false, "", false},
            {nodegraphparams::cfd::PressureTolerance, "Pressure Tolerance", NodeGraphParamType::Float, 1e-4, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(0),
                "-X Boundary Type", "Velocity Inlet",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(0), "-X Velocity X (m/s)", NodeGraphParamType::Float, 0.1, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(0), "-X Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(0), "-X Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(1),
                "+X Boundary Type", "Pressure Outlet",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(1), "+X Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(1), "+X Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(1), "+X Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(2),
                "-Y Boundary Type", "No-Slip Wall",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(2), "-Y Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(2), "-Y Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(2), "-Y Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(3),
                "+Y Boundary Type", "No-Slip Wall",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(3), "+Y Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(3), "+Y Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(3), "+Y Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(4),
                "-Z Boundary Type", "No-Slip Wall",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(4), "-Z Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(4), "-Z Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(4), "-Z Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},

            makeEnumParamDefinition(
                nodegraphparams::cfd::domainBoundaryTypeParam(5),
                "+Z Boundary Type", "No-Slip Wall",
                {"No-Slip Wall", "Velocity Inlet", "Pressure Outlet"}),
            {nodegraphparams::cfd::domainBoundaryVelocityXParam(5), "+Z Velocity X (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityYParam(5), "+Z Velocity Y (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::cfd::domainBoundaryVelocityZParam(5), "+Z Velocity Z (m/s)", NodeGraphParamType::Float, 0.0, 0, false, "", false},
        },
    };
}

static NodeTypeDefinition buildPointsNode() {
    return {
        nodegraphtypes::Points,
        "Points",
        NodeGraphNodeCategory::Geometry,
        {
            makeOutputSocket("Points", NodeGraphValueType::Points, payloadtypes::Points),
        },
        {
            {nodegraphparams::points::PointCount, "Point Count", NodeGraphParamType::Int, 0.0, 1000, false, "", false},
            {nodegraphparams::points::DimX, "Dim X", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::points::DimY, "Dim Y", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::points::DimZ, "Dim Z", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            makeEnumParamDefinition(nodegraphparams::points::Distribution, "Distribution", "Blue Noise", {"Uniform Grid", "Blue Noise"}),
            {nodegraphparams::points::Seed, "Seed", NodeGraphParamType::Int, 0.0, 1, false, "", false},
        },
    };
}

static NodeTypeDefinition buildMeshPointsNode() {
    return {
        nodegraphtypes::MeshPoints,
        "Mesh Points",
        NodeGraphNodeCategory::Geometry,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Remesh}),
            makeOutputSocket("Points", NodeGraphValueType::Points, payloadtypes::Points),
        },
        {},
    };
}

static NodeTypeDefinition buildMergeNode() {
    return {
        nodegraphtypes::Merge,
        "Merge",
        NodeGraphNodeCategory::Meshing,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, true),
            makePolymorphicOutputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, payloadtypes::None),
        },
        {
            {nodegraphparams::merge::MergeDistance, "Merge Distance", NodeGraphParamType::Float, 0.0005, 0, false, "", false},
        },
    };
}

void initNodeGraph(NodeGraphRegistry& registry) {
    payloadtypes::Geometry   = registry.registerPayloadType("geometry", NodeGraphValueType::Mesh);
    payloadtypes::Remesh     = registry.registerPayloadType("remesh",   NodeGraphValueType::Remesh);
    payloadtypes::HeatModel  = registry.registerPayloadType("heat_model", NodeGraphValueType::HeatModel);
    payloadtypes::Fluid      = registry.registerPayloadType("fluid",      NodeGraphValueType::Fluid);
    payloadtypes::Heat       = registry.registerPayloadType("heat",      NodeGraphValueType::Simulation);
    payloadtypes::Voronoi    = registry.registerPayloadType("voronoi",   NodeGraphValueType::Volume);
    payloadtypes::Points     = registry.registerPayloadType("points",    NodeGraphValueType::Points);
    payloadtypes::SerialTemperature = registry.registerPayloadType("serial_temperature", NodeGraphValueType::ScalarFloat);
    payloadtypes::CFD        = registry.registerPayloadType("cfd",       NodeGraphValueType::Simulation);

    registry.registerNodeType(buildModelNode());
    registry.registerNodeType(buildTransformNode());
    registry.registerNodeType(buildGroupNode());
    registry.registerNodeType(buildRemeshNode());
    registry.registerNodeType(buildHeatModelNode());
    registry.registerNodeType(buildFluidNode());
    registry.registerNodeType(buildVoronoiNode());
    registry.registerNodeType(buildCFDNode());
    registry.registerNodeType(buildHeatSolveNode());
    registry.registerNodeType(buildPointsNode());
    registry.registerNodeType(buildMeshPointsNode());
    registry.registerNodeType(buildMergeNode());
    registry.registerNodeType(buildSerialTemperatureNode());
}
