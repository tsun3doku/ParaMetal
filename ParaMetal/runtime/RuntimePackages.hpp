#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "contact/ContactTypes.hpp"
#include "domain/GeometryData.hpp"
#include "domain/ContactData.hpp"
#include "domain/HeatData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"
#include "domain/SerialTemperatureData.hpp"
#include "hash/HashValues.hpp"
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
    ProductHandle productHandle{};
    GeometryData geometry;

    uint64_t computeHash() const { return hashes.geometry; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
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
    ProductHandle productHandle{};
    GeometryData sourceGeometry;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    DisplaySettings display{};
    NodeDataHandle remeshHandle{};
    NodeDataHandle sourceMeshHandle{};
    ProductHandle sourceModelProduct{};

    uint64_t computeHash() const { return hashes.geometry; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
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
    ProductHandle productHandle{};
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
    std::vector<glm::vec4> pointPositions;
    ProductHandle modelProduct{};
    ProductHandle remeshProduct{};

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct PointPackage {
    HashValues hashes{};
    ProductHandle productHandle{};
    NodeDataHandle pointsPayloadHandle{};
    std::vector<glm::vec4> positions;
    uint32_t pointCount = 0;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    uint64_t computeHash() const { return hashes.geometry; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct HeatPackage {
    struct DisplaySettings {
        bool showHeatOverlay = false;
        bool showFluxVectors = false;
        bool showHeatPalette = false;
        float fluxVectorScale = 1.0f;

        bool anyVisible() const {
            return showHeatOverlay || showFluxVectors || showHeatPalette;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    HeatData authored;
    NodeDataHandle heatHandle{};
    DisplaySettings display{};

    std::vector<ProductHandle> remeshProducts;
    std::vector<ProductHandle> modelProducts;
    std::vector<ProductHandle> voronoiProducts;
    std::vector<ProductHandle> contactProducts;
    std::vector<float> resolvedDensity;
    std::vector<float> resolvedSpecificHeat;
    std::vector<float> resolvedConductivity;
    std::vector<float> resolvedInitialTemperaturesC;
    std::vector<uint32_t> resolvedBoundaryConditionTypes;
    std::vector<float> resolvedBoundaryTemperaturesC;
    std::vector<float> resolvedBoundaryHeatFluxes;
    std::vector<float> resolvedBoundaryHeatTransferCoefficients;
    std::vector<float> resolvedVolumetricPowerDensities;
    std::vector<uint64_t> resolvedRobinSourceKeys;
    std::unordered_map<uint64_t, SerialTemperatureData> resolvedSerialSources;

    uint64_t computeHash() const { return hashes.full; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct ContactPackage {
    struct DisplaySettings {
        bool showContactLines = false;

        bool anyVisible() const {
            return showContactLines;
        }
    };

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }

    HashValues hashes{};
    ProductHandle productHandle{};
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
    ProductHandle modelARemeshProduct{};
    ProductHandle modelBRemeshProduct{};
};
