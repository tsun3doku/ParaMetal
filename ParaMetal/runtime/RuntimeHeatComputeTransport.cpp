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

    if (!controller->apply(socketKey, config)) {
        return {};
    }

    if (simGraph) {
        HeatSystem* system = controller->getSystem(socketKey);
        if (system == nullptr) {
            simGraph->unregisterNode(socketKey);
        } else {
            simGraph->registerNode(socketKey, system);
            const uint64_t producerSocket = config.hasAdvection ? package.cfdProduct.outputSocketKey : 0;
            if (producerSocket != 0) {
                SimConnection connection{};
                connection.connectionId = socketKey;
                connection.producerSocket = producerSocket;
                connection.consumerSocket = socketKey;
                connection.mode = SimConnectionMode::Staged;
                simGraph->connect(connection);
            } else {
                simGraph->disconnect(socketKey);
            }
        }
    }

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
    // Deregister before the controller destroys the system the graph points at.
    if (simGraph) {
        simGraph->unregisterNode(socketKey);
    }
    controller->remove(socketKey);
}

bool RuntimeHeatComputeTransport::tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const {
    if (package.models.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.active = package.authored.active;
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
        coupling.faceB = domainProduct->contactCouplingFaceB[cId];
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
    globalDomain.seedToFragment = domainProduct->seedToFragment;
    globalDomain.volumeGridMin = domainProduct->globalVolumeGridMin;
    globalDomain.volumeGridDim = domainProduct->globalVolumeGridDim;
    globalDomain.volumeGridCellSize = domainProduct->globalVolumeGridCellSize;
    globalDomain.nominalCellSize = domainProduct->nominalCellSize;
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
        outConfig.modelVolumetricPowerDensitiesByRuntimeId[runtimeModelId] = model.volumetricPowerDensity;
        outConfig.modelPrescribedTemperatureEnabledByRuntimeId[runtimeModelId] = model.prescribedTemperatureEnabled;
        outConfig.modelPrescribedTemperaturesCByRuntimeId[runtimeModelId] = model.prescribedTemperatureC;
        outConfig.modelDensity[runtimeModelId] = model.density;
        outConfig.modelSpecificHeat[runtimeModelId] = model.specificHeat;
        outConfig.modelConductivity[runtimeModelId] = model.conductivity;
        outConfig.modelLocalToWorldByModelId[runtimeModelId] = model.remeshLocalToWorld;
    }

    if (package.cfdProduct.isValid()) {
        const CFDProduct* cfdProduct = products->resolve<CFDProduct>(package.cfdProduct);
        if (cfdProduct && cfdProduct->isValid()) {
            if (cfdProduct->hashes.geometry == domainProduct->hashes.geometry &&
                cfdProduct->faceFluxCount > 0) {
                outConfig.hasAdvection = true;
                outConfig.advectionFaceFluxBuffer = cfdProduct->faceFluxBuffer;
                outConfig.advectionFaceFluxBufferSize = cfdProduct->faceFluxBufferSize;
                outConfig.advectionBoundaryFluxBuffer = cfdProduct->boundaryFluxBuffer;
                outConfig.advectionBoundaryFluxBufferSize = cfdProduct->boundaryFluxBufferSize;
                outConfig.advectionFaceCount = cfdProduct->faceFluxCount;
                outConfig.advectionBoundaryFaceCount = cfdProduct->boundaryFluxCount;
            }
        }
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
    HashBuilder::combine(structuralHash, static_cast<uint64_t>(package.hasFluid ? 1u : 0u));
    if (package.hasFluid) {
        HashBuilder::combineFloat(structuralHash, package.fluidDensity);
        HashBuilder::combineFloat(structuralHash, package.fluidSpecificHeat);
        HashBuilder::combineFloat(structuralHash, package.fluidConductivity);
    }
    HashBuilder::combine(structuralHash, static_cast<uint64_t>(outConfig.hasAdvection ? 1u : 0u));

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
