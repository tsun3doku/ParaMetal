#include "RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"

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
    if (controller->getCurrentComputeHash(socketKey) == computeHash) {
        return ComputeApplyResult::Unchanged;
    }

    cfd::CFDSystemComputeController::Config config{};
    topology::GlobalDomain tempDomain;
    if (!tryBuildConfig(socketKey, package, config, tempDomain)) {
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
    cfd::CFDSystemComputeController::Config& outConfig,
    topology::GlobalDomain& outDomain) const {

    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return false;
    }
    const VoronoiProduct* domainProduct = products->resolve<VoronoiProduct>(
        package.domainVoronoiProduct);
    if (!domainProduct || !domainProduct->isValid()) {
        return false;
    }

    outConfig = {};
    outConfig.kinematicViscosity = package.fluid.kinematicViscosity;
    outConfig.initialVelocity = package.initialVelocity;
    outConfig.fixedTimeStep = package.fixedTimeStep;
    outConfig.targetCourant = package.targetCourant;
    outConfig.pressureIterations = package.pressureIterations;
    outConfig.pressureTolerance = package.pressureTolerance;
    outConfig.simulationDuration = package.simulationDuration;
    outConfig.computeHash = package.hashes.simulation;
    outConfig.geometryHash = domainProduct->hashes.geometry;

    for (uint32_t i = 0; i < 6; ++i) {
        outConfig.domainBoundaryTypes[i] = package.domainBoundaryTypes[i];
        outConfig.domainBoundaryVelocities[i] = package.domainBoundaryVelocities[i];
    }

    const auto* existingConfig = controller ? controller->getConfig(socketKey) : nullptr;
    if (existingConfig &&
        cfd::CFDSystemComputeController::canUpdateInPlace(*existingConfig, outConfig)) {
        outConfig.domain = nullptr;
        return true;
    }

    topology::GlobalDomain& globalDomain = outDomain;
    globalDomain.fragments.reserve(domainProduct->globalFragmentCount);
    for (uint32_t fragmentId = 0; fragmentId < domainProduct->globalFragmentCount; ++fragmentId) {
        topology::Fragment fragment{};
        fragment.instanceId = domainProduct->fragmentInstanceIds[fragmentId];
        fragment.seedId = domainProduct->fragmentSeedIds[fragmentId];
        fragment.centroid = domainProduct->fragmentCentroids[fragmentId];
        fragment.volume = domainProduct->fragmentVolumes[fragmentId];
        globalDomain.fragments.push_back(fragment);
    }

    globalDomain.faces.reserve(domainProduct->globalFaceCount);
    for (uint32_t faceId = 0; faceId < domainProduct->globalFaceCount; ++faceId) {
        topology::Face face{};
        face.fragmentA = domainProduct->faceFragmentA[faceId];
        face.fragmentB = domainProduct->faceFragmentB[faceId];
        face.areaA = domainProduct->faceAreasA[faceId];
        face.areaVectorA = domainProduct->faceAreaVectorsA[faceId];
        face.momentA = domainProduct->faceMomentsA[faceId];
        face.areaB = domainProduct->faceAreasB[faceId];
        face.areaVectorB = domainProduct->faceAreaVectorsB[faceId];
        face.momentB = domainProduct->faceMomentsB[faceId];
        face.type = static_cast<topology::FaceType>(domainProduct->faceTypes[faceId]);
        face.domainFace = domainProduct->faceDomainFaces[faceId];
        globalDomain.faces.push_back(face);
    }

    globalDomain.boundaryFaces.reserve(domainProduct->globalBoundaryFaceCount);
    for (uint32_t bfId = 0; bfId < domainProduct->globalBoundaryFaceCount; ++bfId) {
        topology::BoundaryFace boundaryFace{};
        boundaryFace.faceIndex = domainProduct->boundaryFaceIndices[bfId];
        boundaryFace.fragment = domainProduct->boundaryFaceFragments[bfId];
        boundaryFace.domainFace = domainProduct->boundaryFaceDomainFaces[bfId];
        globalDomain.boundaryFaces.push_back(boundaryFace);
    }

    globalDomain.instanceFragmentCounts = domainProduct->instanceFragmentCounts;
    globalDomain.seedPositions.reserve(domainProduct->globalSeedPositions.size());
    for (const glm::vec4& position : domainProduct->globalSeedPositions) {
        globalDomain.seedPositions.push_back(glm::vec3(position));
    }
    globalDomain.cellFragmentOffsets = domainProduct->cellFragmentOffsets;
    globalDomain.cellFragments = domainProduct->cellFragments;
    globalDomain.domainMin = domainProduct->globalDomainCorners[0];
    globalDomain.domainMax = domainProduct->globalDomainCorners[0];
    for (const glm::vec3& corner : domainProduct->globalDomainCorners) {
        globalDomain.domainMin = glm::min(globalDomain.domainMin, corner);
        globalDomain.domainMax = glm::max(globalDomain.domainMax, corner);
    }
    globalDomain.volumeGridMin = domainProduct->globalVolumeGridMin;
    globalDomain.volumeGridDim = domainProduct->globalVolumeGridDim;
    globalDomain.volumeGridCellSize = domainProduct->globalVolumeGridCellSize;
    globalDomain.nominalCellSize = domainProduct->nominalCellSize;
    globalDomain.seedIdGridBuffer = domainProduct->globalSeedIdGridBuffer;
    globalDomain.seedIdGridBufferOffset = domainProduct->globalSeedIdGridBufferOffset;

    outConfig.domain = &globalDomain;
    return true;
}
