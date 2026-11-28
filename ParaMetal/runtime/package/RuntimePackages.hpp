#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "heat/HeatProbeConfig.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "voronoi/VoronoiConfig.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashValues.hpp"
#include "runtime/ProductHandle.hpp"
#include "util/Units.hpp"


struct ModelPackage {
    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    std::vector<float> vertexPositions;
    std::vector<uint32_t> triangleIndices;
    std::vector<float> renderPositions;
    std::vector<float> renderNormals;
    std::vector<float> renderTexcoords;
    std::vector<uint32_t> renderIndices;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float canonicalToWorldScale = 1.0f;
    glm::vec3 localBoundsMin{0.0f};
    glm::vec3 localBoundsMax{0.0f};
    glm::vec3 worldBoundsMin{0.0f};
    glm::vec3 worldBoundsMax{0.0f};

    uint64_t computeHash() const { return hashes.geometry; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct GroupPackage {
    HashValues hashes{};
    ProductHandle surfaceProduct{};
    std::vector<uint64_t> displayDependencyKeys;

    uint64_t computeHash() const { return hashes.full; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return surfaceProduct.isValid(); }
};

struct RemeshPackage {
    struct DisplaySettings {
        bool showFaceNormals = false;
        bool showVertexNormals = false;
        float normalLength = 0.05f;

        bool anyVisible() const {
            return showFaceNormals || showVertexNormals;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    std::vector<float> vertexPositions;
    std::vector<uint32_t> triangleIndices;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    DisplaySettings display{};
    ProductHandle sourceModelProduct{};
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

struct VoronoiPackage {
    float canonicalToWorldScale = 1.0f;
    struct DisplaySettings {
        bool showVoronoi = false;
        bool showPoints = false;

        bool anyVisible() const {
            return showVoronoi || showPoints;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    bool active = false;
    int voxelResolution = 128;
    float volumePadding = 0.01f;
    int lloydIterations = int(voronoi::DefaultLloydIterations);
    float lloydVolumeThreshold = voronoi::DefaultLloydVolumeThresholdFraction;
    float lloydDamping = voronoi::DefaultLloydDamping;
    float contactRange = voronoi::DefaultContactRange;
    DisplaySettings display{};
    ProductHandle pointsProduct{};
    std::vector<glm::vec4> pointPositions;
    std::array<glm::vec3, 8> pointDomainCorners{};
    std::vector<ProductHandle> globalRemeshProducts;
    std::vector<ProductHandle> globalModelProducts;
    std::vector<std::array<float, 16>> globalRemeshLocalToWorld;

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct PointPackage {
    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    std::vector<glm::vec4> positions;
    glm::vec3 domainMinimum{0.0f};
    glm::vec3 domainMaximum{0.0f};
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

struct HeatModelPackage {
    uint64_t heatModelKey = 0;
    ProductHandle modelProduct{};
    ProductHandle remeshProduct{};
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::array<float, 16> remeshLocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    float density = 0.0f;
    float specificHeat = 0.0f;
    float conductivity = 0.0f;
    float initialTemperatureC = 0.0f;
    float volumetricPowerDensity = 0.0f;
    bool prescribedTemperatureEnabled = false;
    float prescribedTemperatureC = HeatSimDefaults::ambientTemperatureC;
};

struct HeatPackage {
    struct DisplaySettings {
        bool showHeatOverlay = false;
        bool showFluxVectors = false;
        bool showHeatPalette = false;
        bool showIsocontours = false;
        bool showDomainBox = false;
        bool showFluidVolume = false;
        float fluidVolumeOpacity = 0.25f;
        float fluxVectorScale = 1.0f;
        float isocontourSpacingC = 10.0f;

        bool anyVisible() const {
            return showHeatOverlay || showFluxVectors || showHeatPalette || showIsocontours ||
                showDomainBox || showFluidVolume;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    bool active = false;
    float simulationDuration = 5.0f;
    uint32_t solverMaxIterations = 0;
    float solverConvergenceTolerance = 0.0f;
    units::LengthUnit worldUnit = units::defaultLengthUnit();
    float canonicalToWorldScale = 1.0f;
    DisplaySettings display{};

    std::vector<HeatModelPackage> models;
    bool hasFluid = false;
    float fluidDensity = 0.0f;
    float fluidSpecificHeat = 0.0f;
    float fluidConductivity = 0.0f;
    float fluidInitialTemperatureC = 0.0f;

    ProductHandle domainVoronoiProduct{};
    ProductHandle cfdProduct{};
    std::unordered_map<uint64_t, HeatProbeConfig> resolvedHeatProbes;
    uint64_t probeConfigHash = 0;

    uint32_t domainBoundaryTypes[6] = {};
    float domainBoundaryTemperaturesC[6] = {
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC
    };
    float domainBoundaryHeatFluxes[6] = {};
    float domainBoundaryHeatTransferCoefficients[6] = {};
    float domainBoundaryInflowTemperaturesC[6] = {
        HeatSimDefaults::fluidInitialTemperatureC,
        HeatSimDefaults::fluidInitialTemperatureC,
        HeatSimDefaults::fluidInitialTemperatureC,
        HeatSimDefaults::fluidInitialTemperatureC,
        HeatSimDefaults::fluidInitialTemperatureC,
        HeatSimDefaults::fluidInitialTemperatureC
    };

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct CFDFluidProperties {
    float density = 1.225f;
    float kinematicViscosity = 1.516e-5f;
};

struct CFDPackage {
    struct DisplaySettings {
        bool showStreamlines = false;
        bool showDomainBox = false;
        bool showVelocityPalette = false;
        bool showSlice = false;
        int sliceAxis = 0;
        float slicePosition = 0.5f;
        bool showSliceWireframe = false;

        bool anyVisible() const {
            return showStreamlines || showDomainBox || showVelocityPalette || showSlice;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;

    ProductHandle domainVoronoiProduct{};
    CFDFluidProperties fluid{};
    glm::vec3 initialVelocity{0.0f};

    uint32_t domainBoundaryTypes[6] = {};
    glm::vec3 domainBoundaryVelocities[6] = {};

    float fixedTimeStep = 1.0f / 60.0f;
    float targetCourant = 0.5f;
    int pressureIterations = 0;
    float pressureTolerance = 0.0f;
    float simulationDuration = 10.0f;
    bool active = true;

    float canonicalToWorldScale = 1.0f;
    DisplaySettings display{};

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

template <typename PackageT>
uint64_t packageUpdateHash(const PackageT& package) {
    return package.computeHash();
}

inline uint64_t packageUpdateHash(const HeatPackage& package) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, package.computeHash());
    HashBuilder::combine(hash, package.probeConfigHash);
    return hash;
}
