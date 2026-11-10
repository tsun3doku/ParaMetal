#include "RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"

ComputeApplyResult RuntimeHeatComputeTransport::apply(
    uint64_t socketKey,
    const HeatPackage& package,
    HeatProduct& outProduct) {
    if (!controller || !products || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }
    if (package.models.empty()) {
        return ComputeApplyResult::Failed;
    }

    const uint64_t computeHash = package.hashes.simulation;

    if (controller->getCurrentComputeHash(socketKey) == computeHash) {
        HeatSystemComputeController::Config hostConfig{};
        hostConfig.computeHash = computeHash;
        hostConfig.heatProbesBySourceKey = package.resolvedHeatProbes;
        for (const auto& [sourceKey, data] : package.resolvedSerialSources) {
            hostConfig.serialEnabledBySourceKey[sourceKey] = data.enabled;
            hostConfig.serialPortNamesBySourceKey[sourceKey] = data.portName;
            hostConfig.serialBaudRatesBySourceKey[sourceKey] = data.baudRate;
        }
        if (!controller->apply(socketKey, hostConfig)) {
            return ComputeApplyResult::Failed;
        }
        return ComputeApplyResult::Unchanged;
    }

    HeatSystemComputeController::Config config{};
    if (!tryBuildConfig(socketKey, package, config)) {
        return ComputeApplyResult::Failed;
    }
    config.computeHash = computeHash;

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

uint64_t RuntimeHeatComputeTransport::advectionUpstreamSocket(const HeatPackage& package) const {
    if (!products || !package.cfdProduct.isValid()) {
        return 0;
    }
    if (!products->isCurrent<CFDProduct>(package.cfdProduct)) {
        return 0;
    }
    const CFDProduct* cfdProduct = products->resolve<CFDProduct>(package.cfdProduct);
    if (!cfdProduct || !cfdProduct->isValid() || cfdProduct->faceFluxCount == 0) {
        return 0;
    }
    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return 0;
    }
    const VoronoiProduct* domainProduct = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
    if (!domainProduct || !domainProduct->isValid()) {
        return 0;
    }
    if (cfdProduct->hashes.geometry != domainProduct->hashes.geometry) {
        return 0;
    }
    return package.cfdProduct.outputSocketKey;
}

void RuntimeHeatComputeTransport::commitPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->commitPending(socketKey);
}

void RuntimeHeatComputeTransport::cancelPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->cancelCandidate(socketKey);
}

void RuntimeHeatComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}

bool RuntimeHeatComputeTransport::tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const {
    if (package.models.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.worldUnit = package.worldUnit;
    outConfig.simulationDuration = package.authored.simulationDuration;
    outConfig.solverMaxIterations = package.authored.solverMaxIterations;
    outConfig.solverConvergenceTolerance = package.authored.solverConvergenceTolerance;
    outConfig.hasFluid = package.hasFluid;
    outConfig.fluidDensity = package.fluidDensity;
    outConfig.fluidSpecificHeat = package.fluidSpecificHeat;
    outConfig.fluidConductivity = package.fluidConductivity;
    outConfig.fluidInitialTemperatureC = package.fluidInitialTemperatureC;
    for (const auto& [sourceKey, data] : package.resolvedSerialSources) {
        outConfig.serialEnabledBySourceKey[sourceKey] = data.enabled;
        outConfig.serialPortNamesBySourceKey[sourceKey] = data.portName;
        outConfig.serialBaudRatesBySourceKey[sourceKey] = data.baudRate;
    }
    for (const auto& [sourceKey, data] : package.resolvedHeatProbes) {
        outConfig.heatProbesBySourceKey[sourceKey] = data;
    }

    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return false;
    }
    const VoronoiProduct* domainProduct = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
    if (!domainProduct) {
        return false;
    }
    if (!domainProduct->isValid()) {
        return false;
    }

    topology::GlobalDomain& globalDomain = outConfig.domainVoronoiProduct;
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
    globalDomain.contactCouplings.reserve(domainProduct->globalContactCouplingCount);
    for (uint32_t cId = 0; cId < domainProduct->globalContactCouplingCount; ++cId) {
        topology::ContactCoupling coupling{};
        coupling.faceA = domainProduct->contactCouplingFaceA[cId];
        coupling.fragmentA = domainProduct->contactCouplingFragmentA[cId];
        coupling.faceB = domainProduct->contactCouplingFaceB[cId];
        coupling.fragmentB = domainProduct->contactCouplingFragmentB[cId];
        coupling.couplingArea = domainProduct->contactCouplingAreas[cId];
        globalDomain.contactCouplings.push_back(coupling);
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

    const size_t modelCount = package.models.size();
    outConfig.modelSurfacePositions.reserve(modelCount);
    outConfig.modelSurfaceNormals.reserve(modelCount);
    outConfig.modelSurfaceTriangleIndices.reserve(modelCount);
    outConfig.heatModelKeys.reserve(modelCount);
    outConfig.modelRuntimeModelIds.reserve(modelCount);
    outConfig.modelInitialTemperaturesCByRuntimeId.reserve(modelCount);
    outConfig.modelVolumetricPowerDensitiesByRuntimeId.reserve(modelCount);

    for (size_t i = 0; i < modelCount; ++i) {
        const HeatModelPackage& model = package.models[i];
        if (!products->isCurrent<ModelProduct>(model.modelProduct) ||
            !products->isCurrent<RemeshProduct>(model.remeshProduct)) {
            return false;
        }
        const RemeshProduct* remeshProduct = products->resolve<RemeshProduct>(
            model.remeshProduct);
        if (!remeshProduct ||
            remeshProduct->runtimeModelId == 0) {
            return false;
        }

        const uint32_t runtimeModelId = remeshProduct->runtimeModelId;
        outConfig.modelSurfacePositions.push_back(remeshProduct->surfacePositions);
        outConfig.modelSurfaceNormals.push_back(remeshProduct->surfaceNormals);
        outConfig.modelSurfaceTriangleIndices.push_back(remeshProduct->surfaceTriangleIndices);
        outConfig.heatModelKeys.push_back(model.heatModelKey);
        outConfig.modelRuntimeModelIds.push_back(runtimeModelId);
        outConfig.modelInitialTemperaturesCByRuntimeId[runtimeModelId] = model.initialTemperatureC;
        outConfig.modelVolumetricPowerDensitiesByRuntimeId[runtimeModelId] = model.volumetricPowerDensity;
        outConfig.modelPrescribedTemperatureEnabledByRuntimeId[runtimeModelId] = model.prescribedTemperatureEnabled;
        outConfig.modelPrescribedTemperaturesCByRuntimeId[runtimeModelId] = model.prescribedTemperatureC;
        outConfig.modelDensity[runtimeModelId] = model.density;
        outConfig.modelSpecificHeat[runtimeModelId] = model.specificHeat;
        outConfig.modelConductivity[runtimeModelId] = model.conductivity;
        outConfig.modelLocalToWorldByModelId[runtimeModelId] = model.remeshLocalToWorld;
    }

    if (advectionUpstreamSocket(package) != 0) {
        const CFDProduct* cfdProduct = products->resolve<CFDProduct>(package.cfdProduct);
        outConfig.hasAdvection = true;
        outConfig.advectionFaceFluxBuffer = cfdProduct->faceFluxBuffer;
        outConfig.advectionFaceFluxBufferSize = cfdProduct->faceFluxBufferSize;
        outConfig.advectionBoundaryFluxBuffer = cfdProduct->boundaryFluxBuffer;
        outConfig.advectionBoundaryFluxBufferSize = cfdProduct->boundaryFluxBufferSize;
        outConfig.advectionFaceCount = cfdProduct->faceFluxCount;
        outConfig.advectionBoundaryFaceCount = cfdProduct->boundaryFluxCount;
    }

    for (uint32_t face = 0; face < 6; ++face) {
        outConfig.domainBoundaryTypes[face] = package.domainBoundaryTypes[face];
        outConfig.domainBoundaryTemperaturesC[face] = package.domainBoundaryTemperaturesC[face];
        outConfig.domainBoundaryHeatFluxes[face] = package.domainBoundaryHeatFluxes[face];
        outConfig.domainBoundaryHeatTransferCoefficients[face] = package.domainBoundaryHeatTransferCoefficients[face];
        outConfig.domainBoundaryInflowTemperaturesC[face] = package.domainBoundaryInflowTemperaturesC[face];
    }

    return true;
}
