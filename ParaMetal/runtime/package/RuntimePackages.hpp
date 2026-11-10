#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "domain/GeometryData.hpp"
#include "domain/HeatData.hpp"
#include "domain/FluidData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"
#include "domain/SerialTemperatureData.hpp"
#include "domain/HeatProbeData.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "hash/HashValues.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "nodegraph/NodeGraphProductTypes.hpp"
#include "util/Units.hpp"


struct ModelPackage {
    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;
    NodeDataHandle geometryHandle{};
    GeometryData sourceGeometry;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float canonicalToWorldScale = 1.0f;

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
    std::vector<uint64_t> displayDependencyKeys;
    GeometryData sourceGeometry;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    DisplaySettings display{};
    NodeDataHandle remeshHandle{};
    NodeDataHandle sourceMeshHandle{};
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
    VoronoiData authored;
    NodeDataHandle voronoiHandle{};
    DisplaySettings display{};
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    NodeDataHandle pointsPayloadHandle{};
    std::vector<glm::vec4> pointPositions;
    std::array<glm::vec3, 8> pointDomainCorners{};
    std::vector<NodeDataHandle> globalRemeshHandles;
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
    NodeDataHandle pointsPayloadHandle{};
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
    HeatData authored;
    NodeDataHandle heatHandle{};
    units::LengthUnit worldUnit = units::defaultLengthUnit();
    float canonicalToWorldScale = 1.0f;
    DisplaySettings display{};

    std::vector<HeatModelPackage> models;
    bool hasFluid = false;
    float fluidDensity = 0.0f;
    float fluidSpecificHeat = 0.0f;
    float fluidConductivity = 0.0f;
    float fluidInitialTemperatureC = 0.0f;

    NodeDataHandle domainVoronoiHandle{};
    ProductHandle domainVoronoiProduct{};
    ProductHandle cfdProduct{};
    std::unordered_map<uint64_t, SerialTemperatureData> resolvedSerialSources;
    std::unordered_map<uint64_t, HeatProbeData> resolvedHeatProbes;

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
        FluidData{}.initialTemperatureC,
        FluidData{}.initialTemperatureC,
        FluidData{}.initialTemperatureC,
        FluidData{}.initialTemperatureC,
        FluidData{}.initialTemperatureC,
        FluidData{}.initialTemperatureC
    };

    uint64_t computeHash() const { return hashes.simulation; }
    uint64_t displayHash() const { return hashes.display; }
    bool hasValidProduct() const { return productHandle.isValid(); }
};

struct CFDPackage {
    struct DisplaySettings {
        bool showStreamlines = false;
        bool showDomainBox = false;
        bool showVelocityPalette = false;

        bool anyVisible() const {
            return showStreamlines || showDomainBox || showVelocityPalette;
        }
    };

    HashValues hashes{};
    ProductHandle productHandle{};
    std::vector<uint64_t> displayDependencyKeys;

    ProductHandle domainVoronoiProduct{};
    NodeDataHandle domainVoronoiHandle{};
    NodeDataHandle fluidHandle{};
    NodeDataHandle boundaryHandle{};

    FluidData fluid{};
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

