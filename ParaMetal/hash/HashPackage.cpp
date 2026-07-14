#include "HashPackage.hpp"
#include "HashBuilder.hpp"

#include "runtime/RuntimePackages.hpp"

static void combineHandleHash(uint64_t& hash, const ProductHandle& handle, HashDomain domain) {
    HashBuilder::combine(hash, static_cast<uint64_t>(handle.type));
    HashBuilder::combine(hash, handle.outputSocketKey);
    HashBuilder::combine(hash, handle.hashes.get(domain));
}

void HashPackage::seal(ModelPackage& pkg, const HashValues& geometryHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, geometryHashes.full);
    HashBuilder::combinePod(geometryHash, pkg.geometry.localToWorld);

    pkg.hashes.geometry = geometryHash;
    pkg.hashes.simulation = geometryHash;
    pkg.hashes.full = geometryHash;
    pkg.hashes.display = geometryHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(RemeshPackage& pkg, const HashValues& sourceGeometryHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceGeometryHashes.full);
    HashBuilder::combine(geometryHash, pkg.sourceMeshHandle.key);
    combineHandleHash(geometryHash, pkg.sourceModelProduct, HashDomain::Geometry);
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

void HashPackage::seal(VoronoiPackage& pkg, const HashValues& authoredHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, authoredHashes.full);

    HashBuilder::combine(geometryHash, static_cast<uint64_t>(pkg.domainType));
    HashBuilder::combinePod(geometryHash, pkg.modelLocalToWorld);
    HashBuilder::combine(geometryHash, pkg.pointsPayloadHandle.key);
    HashBuilder::combine(geometryHash, pkg.modelMeshHandle.key);
    HashBuilder::combine(geometryHash, pkg.modelRemeshHandle.key);
    combineHandleHash(geometryHash, pkg.modelProduct, HashDomain::Geometry);
    combineHandleHash(geometryHash, pkg.remeshProduct, HashDomain::Geometry);
    HashBuilder::combinePodVector(geometryHash, pkg.pointPositions);

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

void HashPackage::seal(HeatPackage& pkg, const HashValues& authoredHashes) {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, authoredHashes.simulation);

    HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.remeshProducts.size()));
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.modelProducts.size()));
    for (size_t i = 0; i < pkg.remeshProducts.size(); ++i) {
        combineHandleHash(simulationHash, pkg.remeshProducts[i], HashDomain::Geometry);
        combineHandleHash(simulationHash, pkg.modelProducts[i], HashDomain::Geometry);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedDensity[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedSpecificHeat[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedConductivity[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedInitialTemperaturesC[i]);
        HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.resolvedBoundaryConditionTypes[i]));
        HashBuilder::combineFloat(simulationHash, pkg.resolvedBoundaryTemperaturesC[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedBoundaryHeatFluxes[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedBoundaryHeatTransferCoefficients[i]);
        HashBuilder::combineFloat(simulationHash, pkg.resolvedVolumetricPowerDensities[i]);
    }
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.voronoiProducts.size()));
    for (const ProductHandle& handle : pkg.voronoiProducts) {
        combineHandleHash(simulationHash, handle, HashDomain::Simulation);
    }
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(pkg.contactProducts.size()));
    for (const ProductHandle& handle : pkg.contactProducts) {
        combineHandleHash(simulationHash, handle, HashDomain::Simulation);
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
    HashBuilder::combine(fullHash, authoredHashes.full);

    pkg.hashes.simulation = simulationHash;
    pkg.hashes.geometry = simulationHash;
    pkg.hashes.full = fullHash;
    pkg.hashes.display = displayHash;
    pkg.hashes.thermal = 0;
}

void HashPackage::seal(ContactPackage& pkg, const HashValues& authoredHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, authoredHashes.full);

    HashBuilder::combinePod(geometryHash, pkg.modelALocalToWorld);
    HashBuilder::combinePod(geometryHash, pkg.modelBLocalToWorld);
    combineHandleHash(geometryHash, pkg.modelARemeshProduct, HashDomain::Geometry);
    combineHandleHash(geometryHash, pkg.modelBRemeshProduct, HashDomain::Geometry);

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
