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

    if (package.remeshProducts.empty() || package.voronoiProducts.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.active = package.authored.active;
    outConfig.paused = package.authored.paused;
    outConfig.syntheticDirichletTestEnabled = false;
    outConfig.resetCounter = package.authored.resetCounter;
    outConfig.rewindFrame = package.authored.rewindFrame;
    outConfig.contactThermalConductance = package.authored.contactThermalConductance;
    outConfig.simulationDuration = package.authored.simulationDuration;

    outConfig.modelSurfacePositions.reserve(package.remeshProducts.size());
    outConfig.modelSurfaceNormals.reserve(package.remeshProducts.size());
    outConfig.modelSurfaceTriangleIndices.reserve(package.remeshProducts.size());
    outConfig.modelRuntimeModelIds.reserve(package.remeshProducts.size());
    outConfig.modelInitialTemperaturesCByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelBoundaryConditionTypesByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelBoundaryTemperaturesCByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelBoundaryHeatFluxesByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelBoundaryHeatTransferCoefficientsByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelVolumetricPowerDensitiesByRuntimeId.reserve(package.remeshProducts.size());

    const size_t modelCount = package.remeshProducts.size();
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
        const RemeshProduct* product = products->resolve<RemeshProduct>(package.remeshProducts[i]);
        if (!product || product->runtimeModelId == 0) {
            return false;
        }

        outConfig.modelSurfacePositions.push_back(product->surfacePositions);
        outConfig.modelSurfaceNormals.push_back(product->surfaceNormals);
        outConfig.modelSurfaceTriangleIndices.push_back(product->surfaceTriangleIndices);
        outConfig.modelRuntimeModelIds.push_back(product->runtimeModelId);
        outConfig.modelInitialTemperaturesCByRuntimeId[product->runtimeModelId] = package.resolvedInitialTemperaturesC[i];
        outConfig.modelBoundaryConditionTypesByRuntimeId[product->runtimeModelId] = package.resolvedBoundaryConditionTypes[i];
        outConfig.modelBoundaryTemperaturesCByRuntimeId[product->runtimeModelId] = package.resolvedBoundaryTemperaturesC[i];
        outConfig.modelBoundaryHeatFluxesByRuntimeId[product->runtimeModelId] = package.resolvedBoundaryHeatFluxes[i];
        outConfig.modelBoundaryHeatTransferCoefficientsByRuntimeId[product->runtimeModelId] = package.resolvedBoundaryHeatTransferCoefficients[i];
        outConfig.modelVolumetricPowerDensitiesByRuntimeId[product->runtimeModelId] = package.resolvedVolumetricPowerDensities[i];
        outConfig.modelDensity[product->runtimeModelId] = package.resolvedDensity[i];
        outConfig.modelSpecificHeat[product->runtimeModelId] = package.resolvedSpecificHeat[i];
        outConfig.modelConductivity[product->runtimeModelId] = package.resolvedConductivity[i];

        outConfig.supportingHalfedgeViews[i] = product->supportingHalfedgeView;
        outConfig.supportingAngleViews[i] = product->supportingAngleView;
        outConfig.halfedgeViews[i] = product->halfedgeView;
        outConfig.edgeViews[i] = product->edgeView;
        outConfig.triangleViews[i] = product->triangleView;
        outConfig.lengthViews[i] = product->lengthView;
        outConfig.inputHalfedgeViews[i] = product->inputHalfedgeView;
        outConfig.inputEdgeViews[i] = product->inputEdgeView;
        outConfig.inputTriangleViews[i] = product->inputTriangleView;
        outConfig.inputLengthViews[i] = product->inputLengthView;
    }

    for (const ProductHandle& voronoiProduct : package.voronoiProducts) {
        const VoronoiProduct* product = products->resolve<VoronoiProduct>(voronoiProduct);
        if (!product || !product->isValid()) {
            return false;
        }

        const uint32_t runtimeModelId = product->runtimeModelId;
        if (runtimeModelId == 0) {
            continue;
        }

        bool modelKnown = false;
        for (uint32_t mIdx = 0; mIdx < static_cast<uint32_t>(outConfig.modelRuntimeModelIds.size()); ++mIdx) {
            if (outConfig.modelRuntimeModelIds[mIdx] == runtimeModelId) {
                modelKnown = true;
                break;
            }
        }
        if (!modelKnown) {
            continue;
        }


        outConfig.modelSimNodeBufferByModelId[runtimeModelId] = product->nodeBuffer;
        outConfig.modelSimNodeBufferOffsetByModelId[runtimeModelId] = product->nodeBufferOffset;
        outConfig.modelSimNodeCouplingBufferByModelId[runtimeModelId] = product->couplingBuffer;
        outConfig.modelSimNodeCouplingBufferOffsetByModelId[runtimeModelId] = product->couplingBufferOffset;
        outConfig.simNodeCouplingCounts[runtimeModelId] = product->couplingCount;
        outConfig.simNodeCounts[runtimeModelId] = product->nodeCount;
        outConfig.modelNodePositionsByModelId[runtimeModelId] = product->nodePositions;
        outConfig.modelNodesByModelId[runtimeModelId] = product->nodes;
        outConfig.modelNodeCouplingsByModelId[runtimeModelId] = product->couplings;
        outConfig.modelSurfaceNodeIdsByModelId[runtimeModelId] = product->surfaceNodeIds;
        outConfig.modelSurfacePatchAreasByModelId[runtimeModelId] = product->surfacePatchAreas;
        outConfig.modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = product->gmlsSurfaceStencilBuffer;
        outConfig.modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceStencilBufferOffset;
        outConfig.modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = product->gmlsSurfaceWeightBuffer;
        outConfig.modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceWeightBufferOffset;
        outConfig.modelGMLSSurfaceWeightCountByModelId[runtimeModelId] = product->gmlsSurfaceWeightCount;
        outConfig.modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightBuffer;
        outConfig.modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightBufferOffset;
        outConfig.modelGMLSSurfaceGradientWeightCountByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightCount;
    }

    for (const ProductHandle& contactProduct : package.contactProducts) {
        const ContactProduct* product = products->resolve<ContactProduct>(contactProduct);
        if (!product) {
            return false;
        }

        outConfig.contactCouplings.push_back(product->coupling);
    }

    uint64_t structuralHash = HashBuilder::start();
    HashBuilder::combine(structuralHash, static_cast<uint64_t>(package.remeshProducts.size()));
    for (size_t i = 0; i < package.remeshProducts.size(); ++i) {
        HashBuilder::combine(structuralHash, package.remeshProducts[i].hashes.geometry);
        HashBuilder::combine(structuralHash, package.modelProducts[i].hashes.geometry);
        HashBuilder::combineFloat(structuralHash, package.resolvedDensity[i]);
        HashBuilder::combineFloat(structuralHash, package.resolvedSpecificHeat[i]);
        HashBuilder::combineFloat(structuralHash, package.resolvedConductivity[i]);
        HashBuilder::combineFloat(structuralHash, package.resolvedInitialTemperaturesC[i]);
        HashBuilder::combine(structuralHash, package.resolvedBoundaryConditionTypes[i]);
    }
    for (const ProductHandle& handle : package.voronoiProducts) {
        HashBuilder::combine(structuralHash, handle.hashes.simulation);
    }
    for (const ProductHandle& handle : package.contactProducts) {
        HashBuilder::combine(structuralHash, handle.hashes.simulation);
    }
    HashBuilder::combineFloat(structuralHash, package.authored.contactThermalConductance);
    HashBuilder::combineFloat(structuralHash, package.authored.simulationDuration);
    outConfig.structuralHash = structuralHash;


    return true;
}
