#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/HeatDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

class RuntimeHeatDisplayTransport {
public:
    void setController(HeatDisplayController* updatedController) {
        controller = updatedController;
    }

    void setECSRegistry(ECSRegistry* updatedRegistry) {
        ecsRegistry = updatedRegistry;
    }

    void setVisibleKeys(const std::unordered_set<uint64_t>* keys) {
        visibleKeys = keys;
    }

    void sync(const ECSRegistry& registry) {
        if (!controller) {
            return;
        }

        std::unordered_set<uint64_t> nextSocketKeys;
        auto view = registry.view<HeatPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            if (visibleKeys && visibleKeys->find(socketKey) == visibleKeys->end()) {
                continue;
            }

            const auto& package = registry.get<HeatPackage>(entity);
            HeatDisplayController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                controller->remove(socketKey);
                continue;
            }

            controller->apply(socketKey, config);
            nextSocketKeys.insert(socketKey);
        }

        for (auto entity : registry.view<HeatPackage, Stale>()) {
            controller->remove(static_cast<uint64_t>(entity));
        }

        for (uint64_t socketKey : activeSocketKeys) {
            if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
                controller->remove(socketKey);
            }
        }
        activeSocketKeys = std::move(nextSocketKeys);
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        controller->finalizeSync();
    }

private:
    bool tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatDisplayController::Config& outConfig) const {
        if (!controller || !ecsRegistry || socketKey == 0) {
            return false;
        }
        if (!package.display.anyVisible()) {
            return false;
        }

        const HeatProduct* computeProduct = tryGetProduct<HeatProduct>(*ecsRegistry, socketKey);
        if (!computeProduct || !computeProduct->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.showHeatOverlay = package.display.showHeatOverlay;
        outConfig.showFluxVectors = package.display.showFluxVectors;
        outConfig.showHeatPalette = package.display.showHeatPalette;
        outConfig.fluxVectorScale = package.display.fluxVectorScale;
        outConfig.heatPaletteMinTemp = package.display.heatPaletteMinTemp;
        outConfig.heatPaletteMaxTemp = package.display.heatPaletteMaxTemp;
        outConfig.authoredActive = package.authored.active;
        outConfig.active = computeProduct->active;
        outConfig.paused = computeProduct->paused;

        for (size_t i = 0; i < package.resolvedRemeshHandles.size(); ++i) {
            const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.resolvedModelHandles[i].key);
            const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.resolvedRemeshHandles[i].key);
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

            outConfig.models.push_back(*modelProduct);
            outConfig.modelTemperatures.push_back(package.resolvedInitialTemperature[i]);
            outConfig.modelFixedTemperatures.push_back(package.resolvedFixedTemperatureValues[i]);
            outConfig.modelBoundaryConditions.push_back(static_cast<HeatBoundaryCondition>(package.resolvedBoundaryConditions[i]));
            outConfig.modelBufferViews.push_back(modelBufferViews);

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

        outConfig.displayHash = buildDisplayHash(outConfig, computeProduct->productHash);
        return true;
    }

    HeatDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
