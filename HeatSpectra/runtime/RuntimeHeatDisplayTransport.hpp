#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/HeatDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"

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

        auto view = registry.view<HeatPackage>();
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
    }
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
        if (package.modelProducts.size() != package.models.size()) {
            return false;
        }
        if (package.modelProducts.size() != package.remeshProducts.size()) {
            return false;
        }

        const HeatProduct* computeProduct = tryGetProduct<HeatProduct>(*ecsRegistry, socketKey);
        if (!computeProduct || !computeProduct->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.showHeatOverlay = package.display.showHeatOverlay;
        outConfig.showFluxVectors = package.display.showFluxVectors;
        outConfig.fluxVectorScale = package.display.fluxVectorScale;
        outConfig.authoredActive = package.authored.active;
        outConfig.active = computeProduct->active;
        outConfig.paused = computeProduct->paused;

        for (size_t index = 0; index < package.modelProducts.size(); ++index) {
            const ProductHandle& modelHandle = package.modelProducts[index];
            const ProductHandle& remeshHandle = package.remeshProducts[index];
            const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, modelHandle.outputSocketKey);
            const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, remeshHandle.outputSocketKey);
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
            outConfig.modelTemperatures.push_back(package.models[index].initialTemperature);
            outConfig.modelFixedTemperatures.push_back(package.models[index].fixedTemperatureValue);
            outConfig.modelBoundaryConditions.push_back(package.models[index].boundaryCondition);
            outConfig.modelBufferViews.push_back(modelBufferViews);

            if (modelProduct->runtimeModelId != 0 &&
                index < computeProduct->modelSurfaceBuffers.size() &&
                index < computeProduct->modelSurfaceBufferOffsets.size() &&
                index < computeProduct->modelSurfacePointCounts.size()) {
                outConfig.modelSurfaceBuffers.push_back(computeProduct->modelSurfaceBuffers[index]);
                outConfig.modelSurfaceBufferOffsets.push_back(computeProduct->modelSurfaceBufferOffsets[index]);
                outConfig.modelSurfacePointCounts.push_back(computeProduct->modelSurfacePointCounts[index]);
            } else {
                outConfig.modelSurfaceBuffers.push_back(VK_NULL_HANDLE);
                outConfig.modelSurfaceBufferOffsets.push_back(0);
                outConfig.modelSurfacePointCounts.push_back(0);
            }

            if (modelProduct->runtimeModelId != 0 &&
                index < computeProduct->modelSurfaceGradientBuffers.size() &&
                index < computeProduct->modelSurfaceGradientBufferOffsets.size()) {
                outConfig.modelSurfaceGradientBuffers.push_back(computeProduct->modelSurfaceGradientBuffers[index]);
                outConfig.modelSurfaceGradientBufferOffsets.push_back(computeProduct->modelSurfaceGradientBufferOffsets[index]);
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
};
