#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "contact/ContactTypes.hpp"
#include "domain/GeometryData.hpp"
#include "domain/ContactData.hpp"
#include "domain/HeatData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "nodegraph/NodeGraphProductTypes.hpp"

//                                                   [ Invariant:
//                                                     - Package types are the compiled runtime transport contract
//                                                       consumed by compute and display transports
//                                                     - Packages may combine authored/compiled settings with resolved
//                                                       runtime dependency handles needed by compute or display transports
//                                                     - Packages must not contain resolved runtime Products
//                                                     - Packages are the runtime application boundary ]

struct ModelPackage {
    uint64_t packageHash = 0;
    GeometryData geometry;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    bool matches(const ModelPackage& other) const {
        return packageHash == other.packageHash;
    }
};

struct RemeshPackage {
    struct DisplaySettings {
        bool showRemeshOverlay = false;
        bool showFaceNormals = false;
        bool showVertexNormals = false;
        float normalLength = 0.05f;

        bool anyVisible() const {
            return showRemeshOverlay || showFaceNormals || showVertexNormals;
        }
    };

    uint64_t packageHash = 0;
    GeometryData sourceGeometry;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    DisplaySettings display{};
    NodeDataHandle remeshHandle{};
    NodeDataHandle sourceMeshHandle{};

    bool matches(const RemeshPackage& other) const {
        return packageHash == other.packageHash &&
            display.showRemeshOverlay == other.display.showRemeshOverlay &&
            display.showFaceNormals == other.display.showFaceNormals &&
            display.showVertexNormals == other.display.showVertexNormals &&
            display.normalLength == other.display.normalLength;
    }
};

struct VoronoiPackage {
    struct DisplaySettings {
        bool showVoronoi = false;
        bool showPoints = false;

        bool anyVisible() const {
            return showVoronoi || showPoints;
        }
    };

    uint64_t packageHash = 0;
    VoronoiData authored;
    NodeDataHandle voronoiHandle{};
    DisplaySettings display{};
    std::array<float, 16> modelLocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    DomainType domainType = DomainType::Mesh;
    NodeDataHandle modelMeshHandle{};
    NodeDataHandle modelRemeshHandle{};
    NodeDataHandle pointsPayloadHandle{};

    bool matches(const VoronoiPackage& other) const {
        return packageHash == other.packageHash &&
            display.showVoronoi == other.display.showVoronoi &&
            display.showPoints == other.display.showPoints;
    }
};

struct PointPackage {
    uint64_t packageHash = 0;
    NodeDataHandle pointsPayloadHandle{};
    std::vector<glm::vec4> positions;
    uint32_t pointCount = 0;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    bool matches(const PointPackage& other) const {
        return packageHash == other.packageHash &&
            localToWorld == other.localToWorld;
    }
};

struct HeatPackage {
    struct DisplaySettings {
        bool showHeatOverlay = false;
        bool showFluxVectors = false;
        bool showHeatPalette = false;
        float fluxVectorScale = 1.0f;
        float heatPaletteMinTemp = 0.0f;
        float heatPaletteMaxTemp = 100.0f;

        bool anyVisible() const {
            return showHeatOverlay || showFluxVectors || showHeatPalette;
        }
    };

    uint64_t packageHash = 0;
    HeatData authored;
    NodeDataHandle heatHandle{};
    DisplaySettings display{};

    // Pre-resolved by compiler 
    std::vector<NodeDataHandle> resolvedRemeshHandles;       
    std::vector<NodeDataHandle> resolvedModelHandles;       
    std::vector<float> resolvedDensity;
    std::vector<float> resolvedSpecificHeat;
    std::vector<float> resolvedConductivity;
    std::vector<float> resolvedInitialTemperature;
    std::vector<uint32_t> resolvedBoundaryConditions;
    std::vector<float> resolvedFixedTemperatureValues;

    bool matches(const HeatPackage& other) const {
        return packageHash == other.packageHash &&
            authored.paused == other.authored.paused &&
            authored.resetCounter == other.authored.resetCounter &&
            authored.simulationDuration == other.authored.simulationDuration &&
            display.showHeatOverlay == other.display.showHeatOverlay &&
            display.showFluxVectors == other.display.showFluxVectors &&
            display.showHeatPalette == other.display.showHeatPalette &&
            display.fluxVectorScale == other.display.fluxVectorScale &&
            display.heatPaletteMinTemp == other.display.heatPaletteMinTemp &&
            display.heatPaletteMaxTemp == other.display.heatPaletteMaxTemp;
    }
};

struct ContactPackage {
    struct DisplaySettings {
        bool showContactLines = false;

        bool anyVisible() const {
            return showContactLines;
        }
    };

    uint64_t packageHash = 0;
    ContactData authored;
    NodeDataHandle contactHandle{};
    DisplaySettings display{};
    std::array<float, 16> modelALocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::array<float, 16> modelBLocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    NodeDataHandle modelAMeshHandle{};
    NodeDataHandle modelBMeshHandle{};

    bool matches(const ContactPackage& other) const {
        return packageHash == other.packageHash &&
            display.showContactLines == other.display.showContactLines;
    }
};
