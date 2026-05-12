#pragma once

#include <algorithm>
#include <unordered_set>

#include "contact/ContactTypes.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"

class RuntimeHeatComputeTransport {
public:
    void setController(HeatSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setECSRegistry(ECSRegistry* updatedRegistry) {
        ecsRegistry = updatedRegistry;
    }

    void sync(const ECSRegistry& registry) {
        if (!controller) {
            return;
        }

        std::unordered_set<uint64_t> nextSocketKeys;

        auto view = registry.view<HeatPackage>();
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            const auto& package = registry.get<HeatPackage>(entity);
            auto hashIt = appliedPackageHash.find(socketKey);
            if (hashIt != appliedPackageHash.end() && hashIt->second == package.packageHash) {
                nextSocketKeys.insert(socketKey);
                continue;
            }

            HeatSystemComputeController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                continue;
            }
            controller->configure(socketKey, config);
            nextSocketKeys.insert(socketKey);
        }

        for (uint64_t socketKey : activeSocketKeys) {
            if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
                controller->disable(socketKey);
                appliedPackageHash.erase(socketKey);
            }
        }

        activeSocketKeys = std::move(nextSocketKeys);
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        std::vector<uint64_t> removals;
        auto productView = ecsRegistry->view<HeatProduct>();
        for (auto entity : productView) {
            const uint64_t socketKey = static_cast<uint64_t>(entity);
            if (activeSocketKeys.find(socketKey) == activeSocketKeys.end()) {
                removals.push_back(socketKey);
            }
        }

        for (uint64_t socketKey : removals) {
            removePublishedProduct(socketKey);
        }

        for (uint64_t socketKey : activeSocketKeys) {
            auto entity = static_cast<ECSEntity>(socketKey);
            const auto& package = ecsRegistry->get<HeatPackage>(entity);
            auto hashIt = appliedPackageHash.find(socketKey);
            const HeatProduct* product = tryGetProduct<HeatProduct>(*ecsRegistry, socketKey);
            if (!product || hashIt == appliedPackageHash.end() || hashIt->second != package.packageHash) {
                publishProduct(socketKey);
            }
        }
    }

private:
    bool tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const {
        if (socketKey == 0 || !ecsRegistry) {
            return false;
        }
        if (!package.authored.active) {
            return false;
        }
        if (package.voronoiProducts.empty()) {
            return false;
        }
        if (package.remeshProducts.empty()) {
            return false;
        }

        outConfig = {};
        outConfig.active = package.authored.active;
        outConfig.paused = package.authored.paused;
        outConfig.resetRequested = package.authored.resetRequested;
        outConfig.contactThermalConductance = package.authored.contactThermalConductance;

        // Collect all model payloads from HeatPackage.models
        std::vector<RemeshProduct const*> modelProducts;

        for (size_t i = 0; i < package.remeshProducts.size(); ++i) {
            const ProductHandle& remeshProductHandle = package.remeshProducts[i];
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(*ecsRegistry, remeshProductHandle.outputSocketKey);
            if (!product || product->runtimeModelId == 0) {
                return false;
            }
            modelProducts.push_back(product);
        }

        outConfig.modelIntrinsicMeshes.reserve(modelProducts.size());
        outConfig.modelRuntimeModelIds.reserve(modelProducts.size());
        outConfig.modelTemperatureByRuntimeId.reserve(modelProducts.size());
        outConfig.modelBoundaryConditions.reserve(modelProducts.size());
        outConfig.modelFixedTemperatureValues.reserve(modelProducts.size());

        for (size_t i = 0; i < modelProducts.size(); ++i) {
            const RemeshProduct* product = modelProducts[i];
            const HeatModelData& modelData = package.models[i];
            outConfig.modelIntrinsicMeshes.push_back(product->intrinsicMesh);
            outConfig.modelRuntimeModelIds.push_back(product->runtimeModelId);
            outConfig.modelTemperatureByRuntimeId[product->runtimeModelId] = modelData.initialTemperature;
            outConfig.modelBoundaryConditions[product->runtimeModelId] = static_cast<uint32_t>(modelData.boundaryCondition);
            outConfig.modelFixedTemperatureValues[product->runtimeModelId] = modelData.fixedTemperatureValue;
            outConfig.modelDensity[product->runtimeModelId] = modelData.density;
            outConfig.modelSpecificHeat[product->runtimeModelId] = modelData.specificHeat;
            outConfig.modelConductivity[product->runtimeModelId] = modelData.conductivity;

        }

        // Collect topology buffers from all models
        const size_t modelCount = modelProducts.size();
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
            const RemeshProduct* product = modelProducts[i];
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

        // Collect Voronoi products - each model gets its own independent buffers
        for (const ProductHandle& voronoiProductHandle : package.voronoiProducts) {
            const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(*ecsRegistry, voronoiProductHandle.outputSocketKey);
            if (!product) {
                continue;
            }

            for (const VoronoiSurfaceProduct& surfaceProduct : product->surfaces) {
                const uint32_t runtimeModelId = surfaceProduct.runtimeModelId;
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
                outConfig.modelGMLSInterfaceBufferByModelId[runtimeModelId] = product->gmlsInterfaceBuffer;
                outConfig.modelGMLSInterfaceBufferOffsetByModelId[runtimeModelId] = product->gmlsInterfaceBufferOffset;
                outConfig.modelSeedFlagsBufferByModelId[runtimeModelId] = product->seedFlagsBuffer;
                outConfig.modelSeedFlagsBufferOffsetByModelId[runtimeModelId] = product->seedFlagsBufferOffset;
                outConfig.modelVoronoiNodeCountByModelId[runtimeModelId] = surfaceProduct.nodeCount;
                outConfig.modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceStencilBuffer;
                outConfig.modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceStencilBufferOffset;
                outConfig.modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceWeightBuffer;
                outConfig.modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceWeightBufferOffset;
                outConfig.modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceGradientWeightBuffer;
                outConfig.modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = surfaceProduct.gmlsSurfaceGradientWeightBufferOffset;
                outConfig.modelVoronoiSeedFlagsByModelId[runtimeModelId] = surfaceProduct.seedFlags;
                outConfig.modelVoronoiSeedPositionsByModelId[runtimeModelId] = surfaceProduct.seedPositions;

                
                // Collect surface topology views from VoronoiProduct surfaces
                outConfig.surfaceSupportingHalfedgeViews.push_back(surfaceProduct.supportingHalfedgeView);
                outConfig.surfaceSupportingAngleViews.push_back(surfaceProduct.supportingAngleView);
                outConfig.surfaceHalfedgeViews.push_back(surfaceProduct.halfedgeView);
                outConfig.surfaceEdgeViews.push_back(surfaceProduct.edgeView);
                outConfig.surfaceTriangleViews.push_back(surfaceProduct.triangleView);
                outConfig.surfaceLengthViews.push_back(surfaceProduct.lengthView);
                outConfig.surfaceInputHalfedgeViews.push_back(surfaceProduct.inputHalfedgeView);
                outConfig.surfaceInputEdgeViews.push_back(surfaceProduct.inputEdgeView);
                outConfig.surfaceInputTriangleViews.push_back(surfaceProduct.inputTriangleView);
                outConfig.surfaceInputLengthViews.push_back(surfaceProduct.inputLengthView);
                outConfig.surfaceRuntimeModelIds.push_back(runtimeModelId);
            }
        }

        for (const ProductHandle& contactProductHandle : package.contactProducts) {
            const ContactProduct* product = tryGetProduct<ContactProduct>(*ecsRegistry, contactProductHandle.outputSocketKey);
            if (!product) {
                continue;
            }
            outConfig.contactCouplings.push_back(product->coupling);
        }


        outConfig.computeHash = buildComputeHash(outConfig);
        return true;
    }

    void removePublishedProduct(uint64_t socketKey) {
        if (socketKey == 0) {
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        ecsRegistry->remove<HeatProduct>(entity);
    }

    void publishProduct(uint64_t socketKey) {
        if (!controller || socketKey == 0) {
            return;
        }

        HeatProduct product{};
        if (!controller->exportProduct(socketKey, product)) {
            auto entity = static_cast<ECSEntity>(socketKey);
            ecsRegistry->remove<HeatProduct>(entity);
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<HeatPackage>(entity);
        ecsRegistry->emplace_or_replace<HeatProduct>(entity, product);
        appliedPackageHash[socketKey] = package.packageHash;
    }

    HeatSystemComputeController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    std::unordered_map<uint64_t, uint64_t> appliedPackageHash;
};
