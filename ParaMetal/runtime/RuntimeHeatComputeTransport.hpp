#pragma once

#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "contact/ContactTypes.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProduct.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatSystem.hpp"
#include "heat/HeatSystemComputeController.hpp"
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

        auto view = registry.view<HeatPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            const auto& package = registry.get<HeatPackage>(entity);

            if (!package.authored.active) {
                controller->disable(socketKey);
                removePublishedProduct(socketKey);
                appliedConfigInputHash.erase(socketKey);
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
                removePublishedProduct(socketKey);
                appliedConfigInputHash.erase(socketKey);
            }
        }

        activeSocketKeys = std::move(nextSocketKeys);
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        for (uint64_t socketKey : activeSocketKeys) {
            auto entity = static_cast<ECSEntity>(socketKey);
            const auto& package = ecsRegistry->get<HeatPackage>(entity);
            const uint64_t inputHash = buildConfigInputHash(socketKey, package);
            auto hashIt = appliedConfigInputHash.find(socketKey);
            const HeatProduct* product = tryGetProduct<HeatProduct>(*ecsRegistry, socketKey);
            bool runtimeChanged = !product || product->paused != package.authored.paused || product->resetCounter != package.authored.resetCounter || product->rewindFrame != package.authored.rewindFrame;
            if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash || runtimeChanged) {
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
        if (package.authored.voronoiHandles.empty()) {
            return false;
        }
        if (package.resolvedRemeshHandles.empty()) {
            return false;
        }

        outConfig = {};
        outConfig.active = package.authored.active;
        outConfig.paused = package.authored.paused;
        outConfig.resetCounter = package.authored.resetCounter;
        outConfig.rewindFrame = package.authored.rewindFrame;
        outConfig.contactThermalConductance = package.authored.contactThermalConductance;
        outConfig.modelIntrinsicMeshes.reserve(package.resolvedRemeshHandles.size());
        outConfig.modelRuntimeModelIds.reserve(package.resolvedRemeshHandles.size());
        outConfig.modelTemperatureByRuntimeId.reserve(package.resolvedRemeshHandles.size());
        outConfig.modelBoundaryConditions.reserve(package.resolvedRemeshHandles.size());
        outConfig.modelFixedTemperatureValues.reserve(package.resolvedRemeshHandles.size());

        for (size_t i = 0; i < package.resolvedRemeshHandles.size(); ++i) {
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(*ecsRegistry, package.resolvedRemeshHandles[i].key);
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
        }

        // Collect topology buffers from all models
        const size_t modelCount = package.resolvedRemeshHandles.size();
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
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(*ecsRegistry, package.resolvedRemeshHandles[i].key);
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

        // Collect Voronoi products
        for (const NodeDataHandle& voronoiHandle : package.authored.voronoiHandles) {
            const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(*ecsRegistry, voronoiHandle.key);
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
            outConfig.modelSimGMLSInterfaceBufferByModelId[runtimeModelId] = product->simGMLSInterfaceBuffer;
            outConfig.modelSimGMLSInterfaceBufferOffsetByModelId[runtimeModelId] = product->simGMLSInterfaceBufferOffset;
            outConfig.simGMLSInterfaceCounts[runtimeModelId] = product->simGMLSInterfaceCount;
            outConfig.voronoiNodeCounts[runtimeModelId] = product->nodeCount;
            outConfig.simNodeCounts[runtimeModelId] = product->simNodeCount;
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

        for (const NodeDataHandle& contactHandle : package.authored.contactHandles) {
            const ContactProduct* product = tryGetProduct<ContactProduct>(*ecsRegistry, contactHandle.key);
            if (!product) {
                return false;
            }
            outConfig.contactCouplings.push_back(product->coupling);
        }


        outConfig.simulationDuration = package.authored.simulationDuration;
        outConfig.computeHash = buildConfigInputHash(socketKey, package);
        return true;
    }

    // Structural hash: only mesh/material/topology inputs. Never runtime control state.
    uint64_t buildConfigInputHash(uint64_t socketKey, const HeatPackage& package) const {
        (void)socketKey;
        uint64_t hash = package.hashes.simulation;
        for (size_t i = 0; i < package.resolvedRemeshHandles.size(); ++i) {
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(*ecsRegistry, package.resolvedRemeshHandles[i].key);
            if (!product) {
                return 0;
            }
            HashBuilder::combine(hash, product->hashes.geometry);
        }
        for (const NodeDataHandle& voronoiHandle : package.authored.voronoiHandles) {
            const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(*ecsRegistry, voronoiHandle.key);
            if (!product) {
                return 0;
            }
            HashBuilder::combine(hash, product->hashes.simulation);
        }
        for (const NodeDataHandle& contactHandle : package.authored.contactHandles) {
            const ContactProduct* product = tryGetProduct<ContactProduct>(*ecsRegistry, contactHandle.key);
            if (!product) {
                return 0;
            }
            HashBuilder::combine(hash, product->hashes.simulation);
        }
        return hash;
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
        if (!buildProduct(socketKey, product)) {
            auto entity = static_cast<ECSEntity>(socketKey);
            ecsRegistry->remove<HeatProduct>(entity);
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<HeatPackage>(entity);
        ecsRegistry->emplace_or_replace<HeatProduct>(entity, product);
        appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
    }

    bool buildProduct(uint64_t socketKey, HeatProduct& outProduct) const {
        outProduct = {};
        if (!controller || socketKey == 0) {
            return false;
        }

        const HeatSystem* system = controller->getSystem(socketKey);
        const HeatSystemComputeController::Config* config = controller->getConfig(socketKey);
        if (!system || !config) {
            return false;
        }

        outProduct.active = system->getIsActive();
        outProduct.paused = system->getIsPaused();
        outProduct.resetCounter = system->getResetCounter();
        outProduct.rewindFrame = system->getRewindFrame();
        outProduct.modelRuntimeModelIds.reserve(config->modelRuntimeModelIds.size());
        outProduct.modelSurfaceBuffers.reserve(config->modelRuntimeModelIds.size());
        outProduct.modelSurfaceBufferOffsets.reserve(config->modelRuntimeModelIds.size());
        outProduct.modelSurfacePointCounts.reserve(config->modelRuntimeModelIds.size());
        outProduct.modelSurfaceGradientBuffers.reserve(config->modelRuntimeModelIds.size());
        outProduct.modelSurfaceGradientBufferOffsets.reserve(config->modelRuntimeModelIds.size());

        for (uint32_t runtimeModelId : config->modelRuntimeModelIds) {
            const HeatModelRuntime* heatModel = system->getModelByRuntimeId(runtimeModelId);
            if (!heatModel ||
                runtimeModelId == 0 ||
                heatModel->getSurfaceBuffer() == VK_NULL_HANDLE ||
                heatModel->getIntrinsicVertexCount() == 0) {
                continue;
            }

            outProduct.modelRuntimeModelIds.push_back(runtimeModelId);
            outProduct.modelSurfaceBuffers.push_back(heatModel->getSurfaceBuffer());
            outProduct.modelSurfaceBufferOffsets.push_back(heatModel->getSurfaceBufferOffset());
            outProduct.modelSurfacePointCounts.push_back(static_cast<uint32_t>(heatModel->getIntrinsicVertexCount()));
            outProduct.modelSurfaceGradientBuffers.push_back(heatModel->getSurfaceGradientBuffer());
            outProduct.modelSurfaceGradientBufferOffsets.push_back(heatModel->getSurfaceGradientBufferOffset());
        }

        HashProduct::seal(outProduct);
        return outProduct.isValid();
    }

    HeatSystemComputeController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    std::unordered_map<uint64_t, uint64_t> appliedConfigInputHash;
};
