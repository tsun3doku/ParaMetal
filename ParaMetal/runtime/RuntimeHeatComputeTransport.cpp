#include "RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "hash/HashBuilder.hpp"

#include <iostream>

ProductHandle RuntimeHeatComputeTransport::apply(uint64_t socketKey, const HeatPackage& package) {
    if (!controller || !products || socketKey == 0) {
        return {};
    }

    if (!package.authored.active) {
        remove(socketKey);
        return {};
    }

    HeatSystemComputeController::Config config{};
    if (!tryBuildConfig(socketKey, package, config)) {
        remove(socketKey);
        return {};
    }

    const uint64_t computeHash = package.hashes.simulation;
    config.computeHash = computeHash;

    controller->apply(socketKey, config);

    HeatProduct heatProduct{};
    if (!controller->buildProduct(socketKey, heatProduct)) {
        return {};
    }

    ProductHandle handle = products->publish<HeatProduct>(socketKey, heatProduct);
    return handle;
}

void RuntimeHeatComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}

bool RuntimeHeatComputeTransport::tryBuildConfig(
    uint64_t socketKey,
    const HeatPackage& package,
    HeatSystemComputeController::Config& outConfig) const {

    if (package.models.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.active = package.authored.active;
    outConfig.worldUnit = package.worldUnit;
    outConfig.simulationDuration = package.authored.simulationDuration;
    outConfig.hasFluid = package.hasFluid;
    outConfig.fluidDensity = package.fluidDensity;
    outConfig.fluidSpecificHeat = package.fluidSpecificHeat;
    outConfig.fluidConductivity = package.fluidConductivity;
    outConfig.fluidInitialTemperatureC = package.fluidInitialTemperatureC;
    outConfig.fluidInletTemperatureC = package.fluidInletTemperatureC;
    outConfig.fluidVelocity = package.fluidVelocity;
    for (const auto& [sourceKey, data] : package.resolvedSerialSources) {
        outConfig.serialEnabledBySourceKey[sourceKey] = data.enabled;
        outConfig.serialPortNamesBySourceKey[sourceKey] = data.portName;
        outConfig.serialBaudRatesBySourceKey[sourceKey] = data.baudRate;
    }

    const VoronoiProduct* domainProduct = products->resolve<VoronoiProduct>(
        package.domainVoronoiProduct);
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
        face.area = domainProduct->faceAreas[faceId];
        face.orthogonalDistanceA = domainProduct->faceOrthogonalDistancesA[faceId];
        face.orthogonalDistanceB = domainProduct->faceOrthogonalDistancesB[faceId];
        if (faceId < domainProduct->faceNormals.size()) {
            face.normal = domainProduct->faceNormals[faceId];
        }
        globalDomain.faces.push_back(face);
    }
    globalDomain.contactFaces.reserve(domainProduct->globalContactFaceCount);
    for (uint32_t contactFaceId = 0; contactFaceId < domainProduct->globalContactFaceCount; ++contactFaceId) {
        topology::ContactFace contactFace{};
        contactFace.fragmentA = domainProduct->contactFaceFragmentA[contactFaceId];
        contactFace.fragmentB = domainProduct->contactFaceFragmentB[contactFaceId];
        contactFace.area = domainProduct->contactFaceAreas[contactFaceId];
        if (contactFaceId < domainProduct->contactFaceOrthogonalDistancesA.size()) {
            contactFace.orthogonalDistanceA = domainProduct->contactFaceOrthogonalDistancesA[contactFaceId];
        }
        if (contactFaceId < domainProduct->contactFaceOrthogonalDistancesB.size()) {
            contactFace.orthogonalDistanceB = domainProduct->contactFaceOrthogonalDistancesB[contactFaceId];
        }
        if (contactFaceId < domainProduct->contactFaceGaps.size()) {
            contactFace.gap = domainProduct->contactFaceGaps[contactFaceId];
        }
        globalDomain.contactFaces.push_back(contactFace);
    }
    globalDomain.boundaryFaces.reserve(domainProduct->globalBoundaryFaceCount);
    for (uint32_t bfId = 0; bfId < domainProduct->globalBoundaryFaceCount; ++bfId) {
        topology::BoundaryFace boundaryFace{};
        boundaryFace.fragment = domainProduct->boundaryFaceFragments[bfId];
        boundaryFace.domainFace = domainProduct->boundaryFaceDomainFaces[bfId];
        boundaryFace.area = domainProduct->boundaryFaceAreas[bfId];
        boundaryFace.orthogonalDistance = domainProduct->boundaryFaceOrthogonalDistances[bfId];
        globalDomain.boundaryFaces.push_back(boundaryFace);
    }
    globalDomain.instanceFragmentCounts = domainProduct->instanceFragmentCounts;
    globalDomain.seedPositions.reserve(domainProduct->globalSeedPositions.size());
    for (const glm::vec4& position : domainProduct->globalSeedPositions) {
        globalDomain.seedPositions.push_back(glm::vec3(position));
    }
    globalDomain.sdfGridMin = domainProduct->globalSdfGridMin;
    globalDomain.sdfGridDim = domainProduct->globalSdfGridDim;
    globalDomain.sdfCellSize = domainProduct->globalSdfCellSize;
    globalDomain.nominalCellSize = domainProduct->nominalCellSize;
    globalDomain.sdfValues = domainProduct->globalSdfValues;
    globalDomain.sdfRuntimeModelIds = domainProduct->globalSdfRuntimeModelIds;
    globalDomain.seedIdGridBuffer = domainProduct->globalSeedIdGridBuffer;
    globalDomain.seedIdGridBufferOffset = domainProduct->globalSeedIdGridBufferOffset;
    globalDomain.seedIdGridCount = domainProduct->globalSeedIdGridCount;

    const size_t modelCount = package.models.size();
    outConfig.modelSurfacePositions.reserve(modelCount);
    outConfig.modelSurfaceNormals.reserve(modelCount);
    outConfig.modelSurfaceTriangleIndices.reserve(modelCount);
    outConfig.modelRuntimeModelIds.reserve(modelCount);
    outConfig.modelInitialTemperaturesCByRuntimeId.reserve(modelCount);
    outConfig.modelVolumetricPowerDensitiesByRuntimeId.reserve(modelCount);
    outConfig.supportingHalfedgeViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.supportingAngleViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.halfedgeViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.edgeViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.triangleViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.lengthViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.inputHalfedgeViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.inputEdgeViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.inputTriangleViews.resize(modelCount, VK_NULL_HANDLE);
    outConfig.inputLengthViews.resize(modelCount, VK_NULL_HANDLE);

    for (size_t i = 0; i < modelCount; ++i) {
        const HeatModelPackage& model = package.models[i];
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
        outConfig.modelRuntimeModelIds.push_back(runtimeModelId);
        outConfig.modelInitialTemperaturesCByRuntimeId[runtimeModelId] = model.initialTemperatureC;
        outConfig.modelVolumetricPowerDensitiesByRuntimeId[runtimeModelId] =
            model.volumetricPowerDensity;
        outConfig.modelPrescribedTemperatureEnabledByRuntimeId[runtimeModelId] =
            model.prescribedTemperatureEnabled;
        outConfig.modelPrescribedTemperaturesCByRuntimeId[runtimeModelId] =
            model.prescribedTemperatureC;
        outConfig.modelDensity[runtimeModelId] = model.density;
        outConfig.modelSpecificHeat[runtimeModelId] = model.specificHeat;
        outConfig.modelConductivity[runtimeModelId] = model.conductivity;
        outConfig.modelLocalToWorldByModelId[runtimeModelId] = model.remeshLocalToWorld;

        outConfig.supportingHalfedgeViews[i] = remeshProduct->supportingHalfedgeView;
        outConfig.supportingAngleViews[i] = remeshProduct->supportingAngleView;
        outConfig.halfedgeViews[i] = remeshProduct->halfedgeView;
        outConfig.edgeViews[i] = remeshProduct->edgeView;
        outConfig.triangleViews[i] = remeshProduct->triangleView;
        outConfig.lengthViews[i] = remeshProduct->lengthView;
        outConfig.inputHalfedgeViews[i] = remeshProduct->inputHalfedgeView;
        outConfig.inputEdgeViews[i] = remeshProduct->inputEdgeView;
        outConfig.inputTriangleViews[i] = remeshProduct->inputTriangleView;
        outConfig.inputLengthViews[i] = remeshProduct->inputLengthView;
    }

    uint64_t structuralHash = HashBuilder::start();
    HashBuilder::combine(structuralHash, static_cast<uint64_t>(package.models.size()));
    for (const HeatModelPackage& model : package.models) {
        HashBuilder::combine(structuralHash, model.remeshProduct.hashes.geometry);
        HashBuilder::combine(structuralHash, model.modelProduct.hashes.geometry);
        HashBuilder::combineFloat(structuralHash, model.density);
        HashBuilder::combineFloat(structuralHash, model.specificHeat);
        HashBuilder::combineFloat(structuralHash, model.conductivity);
        HashBuilder::combine(structuralHash, model.prescribedTemperatureEnabled ? 1u : 0u);
    }
    HashBuilder::combine(structuralHash, domainProduct->hashes.geometry);
    HashBuilder::combine(
        structuralHash,
        static_cast<uint64_t>(package.hasFluid ? 1u : 0u));
    if (package.hasFluid) {
        HashBuilder::combineFloat(structuralHash, package.fluidDensity);
        HashBuilder::combineFloat(structuralHash, package.fluidSpecificHeat);
        HashBuilder::combineFloat(structuralHash, package.fluidConductivity);
    }

    for (uint32_t face = 0; face < 6; ++face) {
        outConfig.domainBoundaryTypes[face] = package.domainBoundaryTypes[face];
        outConfig.domainBoundaryTemperaturesC[face] = package.domainBoundaryTemperaturesC[face];
        outConfig.domainBoundaryHeatFluxes[face] = package.domainBoundaryHeatFluxes[face];
        outConfig.domainBoundaryHeatTransferCoefficients[face] = package.domainBoundaryHeatTransferCoefficients[face];

        HashBuilder::combine(structuralHash, static_cast<uint64_t>(package.domainBoundaryTypes[face]));
    }

    outConfig.structuralHash = structuralHash;

    return true;
}
