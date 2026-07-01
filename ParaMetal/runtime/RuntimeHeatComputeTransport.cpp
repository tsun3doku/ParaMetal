#include "RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeProducts.hpp"

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
    outConfig.resetCounter = package.authored.resetCounter;
    outConfig.rewindFrame = package.authored.rewindFrame;
    outConfig.contactThermalConductance = package.authored.contactThermalConductance;
    outConfig.simulationDuration = package.authored.simulationDuration;

    outConfig.modelIntrinsicMeshes.reserve(package.remeshProducts.size());
    outConfig.modelRuntimeModelIds.reserve(package.remeshProducts.size());
    outConfig.modelTemperatureByRuntimeId.reserve(package.remeshProducts.size());
    outConfig.modelBoundaryConditions.reserve(package.remeshProducts.size());
    outConfig.modelFixedTemperatureValues.reserve(package.remeshProducts.size());

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

        outConfig.modelIntrinsicMeshes.push_back(product->intrinsicMesh);
        outConfig.modelRuntimeModelIds.push_back(product->runtimeModelId);
        outConfig.modelTemperatureByRuntimeId[product->runtimeModelId] = package.resolvedInitialTemperature[i];
        outConfig.modelBoundaryConditions[product->runtimeModelId] = package.resolvedBoundaryConditions[i];
        outConfig.modelFixedTemperatureValues[product->runtimeModelId] = package.resolvedFixedTemperatureValues[i];
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


        outConfig.modelVoronoiNodesByModelId[runtimeModelId] = product->mappedVoronoiNodes;
        outConfig.modelVoronoiNodeBufferByModelId[runtimeModelId] = product->nodeBuffer;
        outConfig.modelVoronoiNodeBufferOffsetByModelId[runtimeModelId] = product->nodeBufferOffset;
        outConfig.modelSimNodeBufferByModelId[runtimeModelId] = product->simNodeBuffer;
        outConfig.modelSimNodeBufferOffsetByModelId[runtimeModelId] = product->simNodeBufferOffset;
        outConfig.modelSimNodeBufferSizeByModelId[runtimeModelId] = product->simNodeBufferSize;
        outConfig.modelSimGMLSInterfaceBufferByModelId[runtimeModelId] = product->simGMLSInterfaceBuffer;
        outConfig.modelSimGMLSInterfaceBufferOffsetByModelId[runtimeModelId] = product->simGMLSInterfaceBufferOffset;
        outConfig.simGMLSInterfaceCounts[runtimeModelId] = product->simGMLSInterfaceCount;
        outConfig.voronoiNodeCounts[runtimeModelId] = product->nodeCount;
        outConfig.simNodeCounts[runtimeModelId] = product->simNodeCount;
        outConfig.modelSimNodeVolumesByModelId[runtimeModelId] = product->simNodeVolumes;
        outConfig.modelVoronoiToSimByModelId[runtimeModelId] = product->voronoiToSim;
        outConfig.modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = product->gmlsSurfaceStencilBuffer;
        outConfig.modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceStencilBufferOffset;
        outConfig.modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = product->gmlsSurfaceWeightBuffer;
        outConfig.modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceWeightBufferOffset;
        outConfig.modelGMLSSurfaceWeightCountByModelId[runtimeModelId] = product->gmlsSurfaceWeightCount;
        outConfig.modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightBuffer;
        outConfig.modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightBufferOffset;
        outConfig.modelGMLSSurfaceGradientWeightCountByModelId[runtimeModelId] = product->gmlsSurfaceGradientWeightCount;
        outConfig.modelVoronoiSeedFlagsByModelId[runtimeModelId] = product->seedFlags;
        outConfig.modelVoronoiSeedPositionsByModelId[runtimeModelId] = product->seedPositions;
    }

    for (const ProductHandle& contactProduct : package.contactProducts) {
        const ContactProduct* product = products->resolve<ContactProduct>(contactProduct);
        if (!product) {
            return false;
        }

        outConfig.contactCouplings.push_back(product->coupling);
    }


    return true;
}
