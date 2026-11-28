#include "RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <iostream>

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
    const VoronoiProduct* voronoiProduct = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
    if (!voronoiProduct || !voronoiProduct->isValid()) {
        return 0;
    }
    if (cfdProduct->hashes.geometry != voronoiProduct->hashes.geometry) {
        return 0;
    }
    return package.cfdProduct.outputSocketKey;
}

bool RuntimeHeatComputeTransport::commitPending(uint64_t socketKey) {
    return controller && socketKey != 0 && controller->commitPending(socketKey);
}

void RuntimeHeatComputeTransport::cancelPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->cancelCandidate(socketKey);
}

bool RuntimeHeatComputeTransport::remove(uint64_t socketKey) {
    if (!controller) {
        return false;
    }
    if (socketKey == 0) {
        return true;
    }
    return controller->remove(socketKey);
}

bool RuntimeHeatComputeTransport::hasAnyRetiring() const {
    return controller && controller->hasAnyRetiring();
}

void RuntimeHeatComputeTransport::quiescedClear() {
    if (controller) {
        controller->quiescedClear();
    }
}

bool RuntimeHeatComputeTransport::tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const {
    if (package.models.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.worldUnit = package.worldUnit;
    outConfig.simulationDuration = package.simulationDuration;
    outConfig.solverMaxIterations = package.solverMaxIterations;
    outConfig.solverConvergenceTolerance = package.solverConvergenceTolerance;
    outConfig.hasFluid = package.hasFluid;
    outConfig.fluidDensity = package.fluidDensity;
    outConfig.fluidSpecificHeat = package.fluidSpecificHeat;
    outConfig.fluidConductivity = package.fluidConductivity;
    outConfig.fluidInitialTemperatureC = package.fluidInitialTemperatureC;
    for (const auto& [sourceKey, data] : package.resolvedHeatProbes) {
        outConfig.heatProbesBySourceKey[sourceKey] = data;
    }

    if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
        return false;
    }
    const VoronoiProduct* voronoiProduct = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
    if (!voronoiProduct) {
        return false;
    }
    if (!voronoiProduct->isValid()) {
        return false;
    }
    outConfig.geometryHash = voronoiProduct->hashes.geometry;
    if (!voronoiProduct->hasUniqueSeedToSpaceMapping ||
        voronoiProduct->seedToSpaceIndex == nullptr ||
        voronoiProduct->spaceToFragmentIndex == nullptr) {
        std::cerr << "[RuntimeHeatComputeTransport] Heat requires a unique seed-to-Space mapping for socket "
                  << socketKey << std::endl;
        return false;
    }

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
    outConfig.contactCouplingFaceA = *voronoiProduct->contactCouplingFaceA;
    outConfig.contactCouplingFragmentA = *voronoiProduct->contactCouplingFragmentA;
    outConfig.contactCouplingFaceB = *voronoiProduct->contactCouplingFaceB;
    outConfig.contactCouplingFragmentB = *voronoiProduct->contactCouplingFragmentB;
    outConfig.contactCouplingAreas = *voronoiProduct->contactCouplingAreas;

    outConfig.seedIdGridBuffer = voronoiProduct->globalSeedIdGridBuffer;
    outConfig.seedIdGridBufferOffset = voronoiProduct->globalSeedIdGridBufferOffset;
    outConfig.seedToSpaceIndex = *voronoiProduct->seedToSpaceIndex;
    outConfig.spaceToFragmentIndex = *voronoiProduct->spaceToFragmentIndex;
    outConfig.hasUniqueSeedToSpaceMapping = true;

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

