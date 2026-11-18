#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/HeatDisplayController.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

class RuntimeHeatDisplayTransport {
public:
    void setController(HeatDisplayController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    void sync(const RuntimePackageManager& registry, const std::unordered_set<uint64_t>& visibleKeys) {
        if (!controller) {
            return;
        }

        std::unordered_set<uint64_t> nextSocketKeys;
        for (const auto& [socketKey, package] : registry.heatSystems()) {
            if (!registry.isPackageCurrent(socketKey)) continue;
            if (visibleKeys.find(socketKey) == visibleKeys.end()) {
                continue;
            }

            HeatDisplayController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                controller->remove(socketKey);
                continue;
            }

            controller->apply(socketKey, config);
            nextSocketKeys.insert(socketKey);
        }

        for (uint64_t socketKey : activeSocketKeys) {
            if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
                controller->remove(socketKey);
            }
        }
        activeSocketKeys = nextSocketKeys;
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        controller->finalizeSync();
    }

    void detach(uint64_t socketKey) {
        if (!controller || socketKey == 0) {
            return;
        }
        controller->remove(socketKey);
        activeSocketKeys.erase(socketKey);
    }

    void detachAll() {
        for (uint64_t socketKey : activeSocketKeys) {
            if (controller) {
                controller->remove(socketKey);
            }
        }
        activeSocketKeys.clear();
    }

private:
    bool tryBuildConfig(
        uint64_t socketKey,
        const HeatPackage& package,
        HeatDisplayController::Config& outConfig) const {
        if (!controller || !products || socketKey == 0) {
            return false;
        }

        bool hasProbes = false;
        for (const auto& [nodeId, probeData] : package.resolvedHeatProbes) {
            if (probeData.enabled && !probeData.items.empty()) {
                hasProbes = true;
                break;
            }
        }

        if (!package.display.anyVisible() && !hasProbes) {
            return false;
        }

        if (!products->isCurrent<HeatProduct>(package.productHandle)) {
            return false;
        }
        const HeatProduct* computeProduct = products->resolve<HeatProduct>(package.productHandle);
        if (!computeProduct || !computeProduct->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.showHeatOverlay = package.display.showHeatOverlay;
        outConfig.showFluxVectors = package.display.showFluxVectors;
        outConfig.showHeatPalette = package.display.showHeatPalette;
        outConfig.showIsocontours = package.display.showIsocontours;
        outConfig.showDomainBox = package.display.showDomainBox;
        outConfig.showFluidVolume = package.display.showFluidVolume;
        outConfig.fluidVolumeOpacity = package.display.fluidVolumeOpacity;
        outConfig.fluxVectorScale = package.display.fluxVectorScale;
        outConfig.isocontourSpacingC = package.display.isocontourSpacingC;
        outConfig.authoredActive = package.authored.active;
        outConfig.active = package.authored.active;
        outConfig.canonicalToWorldScale = package.canonicalToWorldScale;
        if (computeProduct->hasFluidVolume) {
            outConfig.fluidTemperatureVolumeView = computeProduct->fluidTemperatureVolumeView;
            outConfig.fluidTemperatureVolumeSampler = computeProduct->fluidTemperatureVolumeSampler;
            outConfig.hasFluidVolume = true;
        }
        for (const auto& [nodeId, probeData] : package.resolvedHeatProbes) {
            if (!probeData.enabled) continue;
            for (const auto& item : probeData.items) {
                if (!item.enabled) continue;
                outConfig.probePositions.push_back(item.worldPosition);
                outConfig.probeIds.push_back(item.id);
                outConfig.probeNames.push_back(item.name.empty() ? ("Probe_" + std::to_string(item.id)) : item.name);
                outConfig.probeShowTemperatures.push_back(probeData.showTemperature);
                outConfig.probeShowNames.push_back(probeData.showName);
                outConfig.probeNodeIds.push_back(nodeId);
            }
        }
        uint64_t productDisplayHash = computeProduct->hashes.display;

        if (package.domainVoronoiProduct.isValid()) {
            if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
                return false;
            }
            const VoronoiProduct* voronoi = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
            if (voronoi && voronoi->isValid()) {
                outConfig.globalDomainCorners = voronoi->globalDomainCorners;
                outConfig.hasGlobalDomain = true;
                outConfig.globalVolumeGridMin = voronoi->globalVolumeGridMin;
                outConfig.globalVolumeGridDim = voronoi->globalVolumeGridDim;
                outConfig.globalVolumeGridCellSize = voronoi->globalVolumeGridCellSize;
                HashBuilder::combine(productDisplayHash, voronoi->hashes.geometry);
                HashBuilder::combine(productDisplayHash, voronoi->hashes.display);
            }
        }

        for (const HeatModelPackage& model : package.models) {
            if (!products->isCurrent<ModelProduct>(model.modelProduct) ||
                !products->isCurrent<RemeshProduct>(model.remeshProduct)) {
                return false;
            }
            const ModelProduct* modelProduct = products->resolve<ModelProduct>(
                model.modelProduct);
            const RemeshProduct* remeshProduct = products->resolve<RemeshProduct>(
                model.remeshProduct);
            if (!remeshProduct || !modelProduct || modelProduct->runtimeModelId == 0) {
                return false;
            }

            std::array<VkBufferView, 11> modelBufferViews = {
                remeshProduct->supportingHalfedgeView,
                remeshProduct->supportingAngleView,
                remeshProduct->halfedgeView,
                remeshProduct->edgeView,
                remeshProduct->triangleView,
                remeshProduct->lengthView,
                remeshProduct->inputHalfedgeView,
                remeshProduct->inputEdgeView,
                remeshProduct->inputTriangleView,
                remeshProduct->inputLengthView,
                VK_NULL_HANDLE
            };
            bool modelValid = true;
            for (int vIdx = 0; vIdx < 10; ++vIdx) {
                if (modelBufferViews[vIdx] == VK_NULL_HANDLE) {
                    modelValid = false;
                    break;
                }
            }
            if (!modelValid) {
                return false;
            }

            outConfig.modelRuntimeIds.push_back(modelProduct->runtimeModelId);
            outConfig.modelRenderVertexBuffers.push_back(modelProduct->renderVertexBuffer);
            outConfig.modelRenderVertexBufferOffsets.push_back(modelProduct->renderVertexBufferOffset);
            outConfig.modelRenderIndexBuffers.push_back(modelProduct->renderIndexBuffer);
            outConfig.modelRenderIndexBufferOffsets.push_back(modelProduct->renderIndexBufferOffset);
            outConfig.modelRenderIndexCounts.push_back(modelProduct->renderIndexCount);
            outConfig.modelMatrices.push_back(toMat4(model.localToWorld));
            outConfig.modelInitialTemperaturesC.push_back(model.initialTemperatureC);
            outConfig.modelBufferViews.push_back(modelBufferViews);
            HashBuilder::combine(productDisplayHash, modelProduct->hashes.display);

            const auto surfaceIt = std::find(
                computeProduct->modelRuntimeModelIds.begin(),
                computeProduct->modelRuntimeModelIds.end(),
                modelProduct->runtimeModelId);
            const size_t surfaceIndex = surfaceIt != computeProduct->modelRuntimeModelIds.end()
                ? static_cast<size_t>(std::distance(computeProduct->modelRuntimeModelIds.begin(), surfaceIt))
                : static_cast<size_t>(-1);

            if (surfaceIndex != static_cast<size_t>(-1) &&
                surfaceIndex < computeProduct->modelSurfaceBuffers.size() &&
                surfaceIndex < computeProduct->modelSurfaceBufferOffsets.size() &&
                surfaceIndex < computeProduct->modelSurfacePointCounts.size()) {
                outConfig.modelSurfaceBuffers.push_back(computeProduct->modelSurfaceBuffers[surfaceIndex]);
                outConfig.modelSurfaceBufferOffsets.push_back(computeProduct->modelSurfaceBufferOffsets[surfaceIndex]);
                outConfig.modelSurfacePointCounts.push_back(computeProduct->modelSurfacePointCounts[surfaceIndex]);
            } else {
                outConfig.modelSurfaceBuffers.push_back(VK_NULL_HANDLE);
                outConfig.modelSurfaceBufferOffsets.push_back(0);
                outConfig.modelSurfacePointCounts.push_back(0);
            }

            if (surfaceIndex != static_cast<size_t>(-1) &&
                surfaceIndex < computeProduct->modelSurfaceGradientBuffers.size() &&
                surfaceIndex < computeProduct->modelSurfaceGradientBufferOffsets.size()) {
                outConfig.modelSurfaceGradientBuffers.push_back(computeProduct->modelSurfaceGradientBuffers[surfaceIndex]);
                outConfig.modelSurfaceGradientBufferOffsets.push_back(computeProduct->modelSurfaceGradientBufferOffsets[surfaceIndex]);
            } else {
                outConfig.modelSurfaceGradientBuffers.push_back(VK_NULL_HANDLE);
                outConfig.modelSurfaceGradientBufferOffsets.push_back(0);
            }
        }

        outConfig.displayHash = buildDisplayHash(outConfig, productDisplayHash);
        HashBuilder::combine(outConfig.displayHash, package.productHandle.publicationId);
        return true;
    }

    HeatDisplayController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
