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
#include "hash/HashValues.hpp"
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
    HashValues hashes{};
    GeometryData geometry;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    bool matches(const ModelPackage& other) const {
        return hashes.full == other.hashes.full;
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

    HashValues hashes{};
    GeometryData sourceGeometry;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    DisplaySettings display{};
    NodeDataHandle remeshHandle{};
    NodeDataHandle sourceMeshHandle{};

    bool matches(const RemeshPackage& other) const {
        return hashes.full == other.hashes.full;
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

    HashValues hashes{};
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
        return hashes.full == other.hashes.full;
    }
};

struct PointPackage {
    HashValues hashes{};
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
        return hashes.full == other.hashes.full;
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

    HashValues hashes{};
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
        return hashes.full == other.hashes.full;
    }
};

struct ContactPackage {
    struct DisplaySettings {
        bool showContactLines = false;

        bool anyVisible() const {
            return showContactLines;
        }
    };

    HashValues hashes{};
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
        return hashes.full == other.hashes.full;
    }
};
