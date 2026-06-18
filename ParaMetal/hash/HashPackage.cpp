#include "HashPackage.hpp"
#include "HashBuilder.hpp"

#include "runtime/RuntimePackages.hpp"

void HashPackage::seal(ModelPackage& pkg) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, pkg.geometry.hashes.full);
    HashBuilder::combinePod(geometryHash, pkg.localToWorld);

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = geometryHash;
    pkg.hashes.display = geometryHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(RemeshPackage& pkg) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, pkg.sourceGeometry.hashes.full);
    HashBuilder::combine(geometryHash, pkg.sourceMeshHandle.key);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(pkg.iterations));
    HashBuilder::combineFloat(geometryHash, pkg.minAngleDegrees);
    HashBuilder::combineFloat(geometryHash, pkg.maxEdgeLength);
    HashBuilder::combineFloat(geometryHash, pkg.stepSize);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showRemeshOverlay ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showFaceNormals ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showVertexNormals ? 1u : 0u));
    HashBuilder::combineFloat(displayHash, pkg.display.normalLength);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = fullHash;
    pkg.hashes.display = displayHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(VoronoiPackage& pkg) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, pkg.authored.hashes.full);

    HashBuilder::combine(geometryHash, static_cast<uint64_t>(pkg.domainType));
    HashBuilder::combinePod(geometryHash, pkg.modelLocalToWorld);
    HashBuilder::combine(geometryHash, pkg.pointsPayloadHandle.key);
    HashBuilder::combine(geometryHash, pkg.modelMeshHandle.key);
    HashBuilder::combine(geometryHash, pkg.modelRemeshHandle.key);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showVoronoi ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showPoints ? 1u : 0u));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = fullHash;
    pkg.hashes.display = displayHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(PointPackage& pkg) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, pkg.pointsPayloadHandle.key);
    HashBuilder::combinePodVector(geometryHash, pkg.positions);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(pkg.pointCount));
    for (float v : pkg.localToWorld) {
        HashBuilder::combineFloat(geometryHash, v);
    }

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = geometryHash;
    pkg.hashes.display = geometryHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(HeatPackage& pkg) {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, pkg.authored.hashes.simulation);

    HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.resolvedRemeshHandles.size()));
    for (size_t i = 0; i < pkg.resolvedRemeshHandles.size(); ++i) {
        HashBuilder::combine(simulationHash, pkg.resolvedRemeshHandles[i].key);
        HashBuilder::combine(simulationHash, pkg.resolvedModelHandles[i].key);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedDensity[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedSpecificHeat[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedConductivity[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedInitialTemperature[i]);
        HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.resolvedBoundaryConditions[i]));
        HashBuilder::combineFloat(simulationHash, pkg.resolvedFixedTemperatureValues[i]);
    }

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showHeatOverlay ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showFluxVectors ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showHeatPalette ? 1u : 0u));
    HashBuilder::combineFloat(displayHash, pkg.display.fluxVectorScale);
    HashBuilder::combineFloat(displayHash, pkg.display.heatPaletteMinTemp);
    HashBuilder::combineFloat(displayHash, pkg.display.heatPaletteMaxTemp);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, displayHash);
    HashBuilder::combine(fullHash, pkg.authored.hashes.full);

    pkg.hashes.simulation = simulationHash;
    pkg.hashes.geometry = simulationHash;
    pkg.hashes.full = fullHash;
    pkg.hashes.display = displayHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(ContactPackage& pkg) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, pkg.authored.hashes.full);

    HashBuilder::combinePod(geometryHash, pkg.modelALocalToWorld);
    HashBuilder::combinePod(geometryHash, pkg.modelBLocalToWorld);
    HashBuilder::combine(geometryHash, pkg.modelAMeshHandle.key);
    HashBuilder::combine(geometryHash, pkg.modelBMeshHandle.key);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(pkg.display.showContactLines ? 1u : 0u));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = fullHash;
    pkg.hashes.display = displayHash;
    pkg.hashes.thermal = 0;
}
