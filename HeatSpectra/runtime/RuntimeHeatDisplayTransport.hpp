#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/HeatDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"

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
        if (package.sourceModelProducts.size() != package.sourceTemperatures.size()) {
            return false;
        }
        if (package.receiverModelProducts.size() != package.receiverRemeshProducts.size()) {
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
        // Get surfaceBufferView from VoronoiProduct (Voronoi owns the surface buffers and views)
        std::unordered_map<uint32_t, VkBufferView> surfaceBufferViewByRuntimeModelId;
        if (package.voronoiProduct.isValid()) {
            const VoronoiProduct* voronoiProduct = tryGetProduct<VoronoiProduct>(*ecsRegistry, package.voronoiProduct.outputSocketKey);
            if (voronoiProduct) {
                surfaceBufferViewByRuntimeModelId.reserve(voronoiProduct->surfaces.size());
                for (const VoronoiSurfaceProduct& surfaceProduct : voronoiProduct->surfaces) {
                    surfaceBufferViewByRuntimeModelId[surfaceProduct.runtimeModelId] = surfaceProduct.surfaceBufferView;
                }
            }
        }

        for (size_t index = 0; index < package.sourceModelProducts.size(); ++index) {
            const ProductHandle& modelHandle = package.sourceModelProducts[index];
            const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, modelHandle.outputSocketKey);
            if (!modelProduct || modelProduct->runtimeModelId == 0) {
                return false;
            }

            outConfig.sourceModels.push_back(*modelProduct);
            outConfig.sourceTemperatures.push_back(package.sourceTemperatures[index]);
        }

        for (size_t index = 0; index < package.receiverModelProducts.size(); ++index) {
            const ProductHandle& modelHandle = package.receiverModelProducts[index];
            const ProductHandle& remeshHandle = package.receiverRemeshProducts[index];
            const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, modelHandle.outputSocketKey);
            const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, remeshHandle.outputSocketKey);
            if (!remeshProduct || !modelProduct || modelProduct->runtimeModelId == 0) {
                return false;
            }

            const auto surfaceBufferViewIt = surfaceBufferViewByRuntimeModelId.find(modelProduct->runtimeModelId);
            std::array<VkBufferView, 11> receiverBufferViews = {
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
                surfaceBufferViewIt != surfaceBufferViewByRuntimeModelId.end()
                    ? surfaceBufferViewIt->second
                    : VK_NULL_HANDLE
            };
            bool receiverValid = true;
            for (VkBufferView bufferView : receiverBufferViews) {
                if (bufferView == VK_NULL_HANDLE) {
                    receiverValid = false;
                    break;
                }
            }
            if (!receiverValid) {
                return false;
            }

            outConfig.receiverModels.push_back(*modelProduct);
            outConfig.receiverBufferViews.push_back(receiverBufferViews);

            if (modelProduct->runtimeModelId != 0 &&
                index < computeProduct->receiverSurfaceBuffers.size() &&
                index < computeProduct->receiverSurfaceBufferOffsets.size() &&
                index < computeProduct->receiverSurfacePointCounts.size()) {
                outConfig.receiverSurfaceBuffers.push_back(computeProduct->receiverSurfaceBuffers[index]);
                outConfig.receiverSurfaceBufferOffsets.push_back(computeProduct->receiverSurfaceBufferOffsets[index]);
                outConfig.receiverSurfacePointCounts.push_back(computeProduct->receiverSurfacePointCounts[index]);
            } else {
                outConfig.receiverSurfaceBuffers.push_back(VK_NULL_HANDLE);
                outConfig.receiverSurfaceBufferOffsets.push_back(0);
                outConfig.receiverSurfacePointCounts.push_back(0);
            }

            if (modelProduct->runtimeModelId != 0 &&
                index < computeProduct->receiverSurfaceGradientBuffers.size() &&
                index < computeProduct->receiverSurfaceGradientBufferOffsets.size()) {
                outConfig.receiverSurfaceGradientBuffers.push_back(computeProduct->receiverSurfaceGradientBuffers[index]);
                outConfig.receiverSurfaceGradientBufferOffsets.push_back(computeProduct->receiverSurfaceGradientBufferOffsets[index]);
            } else {
                outConfig.receiverSurfaceGradientBuffers.push_back(VK_NULL_HANDLE);
                outConfig.receiverSurfaceGradientBufferOffsets.push_back(0);
            }
        }

        outConfig.displayHash = buildDisplayHash(outConfig, computeProduct->productHash);
        return true;
    }

    HeatDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
};
