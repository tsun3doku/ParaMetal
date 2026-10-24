#include "RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeProducts.hpp"
#include <iostream>

ProductHandle RuntimeCFDComputeTransport::apply(uint64_t socketKey, const CFDPackage& package) {
    if (!controller || !products || socketKey == 0) {
        return {};
    }

    if (!package.active) {
        remove(socketKey);
        return {};
    }

    cfd::CFDSystemComputeController::Config config{};
    topology::GlobalDomain tempDomain;
    if (!tryBuildConfig(socketKey, package, config, tempDomain)) {
        remove(socketKey);
        return {};
    }

    if (!controller->apply(socketKey, config)) {
        // Replacement failed, previous generation remains active and publish is skipped
        return {};
    }

    if (simGraph) {
        cfd::CFDSystem* system = controller->getSystem(socketKey);
        if (system != nullptr) {
            simGraph->registerNode(socketKey, system);
        } else {
            simGraph->unregisterNode(socketKey);
        }
    }

    CFDProduct cfdProduct{};
    if (!controller->buildProduct(socketKey, cfdProduct)) {
        return {};
    }

    return products->publish<CFDProduct>(socketKey, cfdProduct);
}

void RuntimeCFDComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    // Deregister before the controller destroys the system the graph points at
    if (simGraph) {
        simGraph->unregisterNode(socketKey);
    }
    controller->remove(socketKey);
}

bool RuntimeCFDComputeTransport::tryBuildConfig(
    uint64_t socketKey,
    const CFDPackage& package,
    cfd::CFDSystemComputeController::Config& outConfig,
    topology::GlobalDomain& outDomain) const {

    const VoronoiProduct* domainProduct = products->resolve<VoronoiProduct>(
        package.domainVoronoiProduct);
    if (!domainProduct || !domainProduct->isValid()) {
        return false;
    }

    outConfig = {};
    outConfig.active = package.active;
    outConfig.kinematicViscosity = package.fluid.kinematicViscosity;
    outConfig.initialVelocity = package.initialVelocity;
    outConfig.fixedTimeStep = package.fixedTimeStep;
    outConfig.targetCourant = package.targetCourant;
    outConfig.pressureIterations = package.pressureIterations;
    outConfig.pressureTolerance = package.pressureTolerance;
    outConfig.computeHash = package.hashes.simulation;
    outConfig.geometryHash = domainProduct->hashes.geometry;

    for (uint32_t i = 0; i < 6; ++i) {
        outConfig.domainBoundaryTypes[i] = package.domainBoundaryTypes[i];
        outConfig.domainBoundaryVelocities[i] = package.domainBoundaryVelocities[i];
    }

    // Skip domain only when update is guaranteed in-place, replacement requires domain
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

    outConfig.domain = &globalDomain;
    return true;
}
