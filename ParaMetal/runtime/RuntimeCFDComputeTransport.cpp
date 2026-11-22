#include "RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <iostream>

ComputeApplyResult RuntimeCFDComputeTransport::apply(
    uint64_t socketKey,
    const CFDPackage& package,
    CFDProduct& outProduct) {
    if (!controller || !products || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }

    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return ComputeApplyResult::Failed;
    }

    const uint64_t computeHash = package.hashes.simulation;
    const VoronoiProduct* voronoiProduct = products->resolve<VoronoiProduct>(
        package.domainVoronoiProduct);
    if (voronoiProduct &&
        controller->getCurrentComputeHash(socketKey) == computeHash &&
        controller->getCurrentGeometryHash(socketKey) == voronoiProduct->hashes.geometry) {
        return ComputeApplyResult::Unchanged;
    }

    cfd::CFDSystemComputeController::Config config{};
    if (!tryBuildConfig(socketKey, package, config)) {
        return ComputeApplyResult::Failed;
    }

    if (!controller->apply(socketKey, config)) {
        return ComputeApplyResult::Failed;
    }

    outProduct = {};
    if (!controller->buildProduct(socketKey, outProduct)) {
        controller->cancelCandidate(socketKey);
        return ComputeApplyResult::Failed;
    }
    return ComputeApplyResult::Prepared;
}

void RuntimeCFDComputeTransport::commitPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->commitPending(socketKey);
}

void RuntimeCFDComputeTransport::cancelPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->cancelCandidate(socketKey);
}

void RuntimeCFDComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}

bool RuntimeCFDComputeTransport::tryBuildConfig(
    uint64_t socketKey,
    const CFDPackage& package,
    cfd::CFDSystemComputeController::Config& outConfig) const {

    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return false;
    }
    const VoronoiProduct* voronoiProduct = products->resolve<VoronoiProduct>(
        package.domainVoronoiProduct);
    if (!voronoiProduct || !voronoiProduct->isValid()) {
        return false;
    }
    if (!voronoiProduct->hasUniqueSeedToSpaceMapping ||
        voronoiProduct->seedToSpaceIndex == nullptr ||
        voronoiProduct->spaceToFragmentIndex == nullptr) {
        std::cerr << "[RuntimeCFDComputeTransport] CFD requires a unique seed-to-Space mapping for socket "
                  << socketKey << std::endl;
        return false;
    }

    outConfig = {};
    outConfig.fragmentInstanceIds = *voronoiProduct->fragmentInstanceIds;
    outConfig.fragmentSeedIds = *voronoiProduct->fragmentSeedIds;
    outConfig.fragmentCentroids = *voronoiProduct->fragmentCentroids;
    outConfig.fragmentVolumes = *voronoiProduct->fragmentVolumes;
    outConfig.faceFragmentA = *voronoiProduct->faceFragmentA;
    outConfig.faceFragmentB = *voronoiProduct->faceFragmentB;
    outConfig.faceAreasA = *voronoiProduct->faceAreasA;
    outConfig.faceAreasB = *voronoiProduct->faceAreasB;
    outConfig.faceAreaVectorsA = *voronoiProduct->faceAreaVectorsA;
    outConfig.faceAreaVectorsB = *voronoiProduct->faceAreaVectorsB;
    outConfig.faceMomentsA = *voronoiProduct->faceMomentsA;
    outConfig.faceMomentsB = *voronoiProduct->faceMomentsB;
    outConfig.faceTypes = *voronoiProduct->faceTypes;
    outConfig.faceDomainFaces = *voronoiProduct->faceDomainFaces;
    outConfig.boundaryFaceIndices = *voronoiProduct->boundaryFaceIndices;
    outConfig.faceLoopOffsets = *voronoiProduct->faceLoopOffsets;
    outConfig.loopVertexOffsets = *voronoiProduct->loopVertexOffsets;
    outConfig.loopVertices = *voronoiProduct->loopVertices;
    outConfig.domainMin = voronoiProduct->domainMin;
    outConfig.domainMax = voronoiProduct->domainMax;
    outConfig.volumeGridMin = voronoiProduct->volumeGridMin;
    outConfig.volumeGridDim = voronoiProduct->volumeGridDim;
    outConfig.volumeGridCellSize = voronoiProduct->volumeGridCellSize;

    outConfig.seedIdGridBuffer = voronoiProduct->globalSeedIdGridBuffer;
    outConfig.seedIdGridBufferOffset = voronoiProduct->globalSeedIdGridBufferOffset;
    outConfig.seedToSpaceIndex = *voronoiProduct->seedToSpaceIndex;
    outConfig.spaceToFragmentIndex = *voronoiProduct->spaceToFragmentIndex;
    outConfig.spaceFragmentCount = voronoiProduct->spaceFragmentCount;
    outConfig.hasUniqueSeedToSpaceMapping = true;
    outConfig.kinematicViscosity = package.fluid.kinematicViscosity;
    outConfig.initialVelocity = package.initialVelocity;
    outConfig.fixedTimeStep = package.fixedTimeStep;
    outConfig.targetCourant = package.targetCourant;
    outConfig.pressureIterations = package.pressureIterations;
    outConfig.pressureTolerance = package.pressureTolerance;
    outConfig.simulationDuration = package.simulationDuration;
    outConfig.computeHash = package.hashes.simulation;
    outConfig.geometryHash = voronoiProduct->hashes.geometry;

    for (uint32_t i = 0; i < 6; ++i) {
        outConfig.domainBoundaryTypes[i] = package.domainBoundaryTypes[i];
        outConfig.domainBoundaryVelocities[i] = package.domainBoundaryVelocities[i];
    }
    return true;
}

