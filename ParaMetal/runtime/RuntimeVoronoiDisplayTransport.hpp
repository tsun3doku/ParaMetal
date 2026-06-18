#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/VoronoiDisplayController.hpp"

#include <iostream>
#include <unordered_set>
#include <vector>

class RuntimeVoronoiDisplayTransport {
public:
    void setController(VoronoiDisplayController* updatedController) {
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
        auto view = registry.view<VoronoiPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            if (visibleKeys && visibleKeys->find(socketKey) == visibleKeys->end()) {
                continue;
            }

            const auto& package = registry.get<VoronoiPackage>(entity);
            VoronoiDisplayController::Config config{};
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
        activeSocketKeys = std::move(nextSocketKeys);
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        controller->finalizeSync();
    }

private:
    bool tryBuildConfig(uint64_t socketKey, const VoronoiPackage& package, VoronoiDisplayController::Config& outConfig) const {
        if (!controller || !ecsRegistry || socketKey == 0) {
            return false;
        }
        if (!package.display.showVoronoi && !package.display.showPoints) {
            return false;
        }

        const VoronoiProduct* computeProduct = tryGetProduct<VoronoiProduct>(*ecsRegistry, socketKey);
        if (!computeProduct) {
            std::cerr << "[VoronoiDisplayTransport] no computeProduct for socketKey=" << socketKey << std::endl;
            return false;
        }
        if (!computeProduct->isValid()) {
            std::cerr << "[VoronoiDisplayTransport] computeProduct invalid: nodeCount=" << computeProduct->nodeCount
                      << " simNodeCount=" << computeProduct->simNodeCount
                      << " nodeBuffer=" << computeProduct->nodeBuffer
                      << " simNodeBuffer=" << computeProduct->simNodeBuffer
                      << " neighborIndicesBuffer=" << computeProduct->voronoiNeighborIndicesBuffer
                      << " seedPositionBuffer=" << computeProduct->seedPositionBuffer
                      << " candidateBuffer=" << computeProduct->candidateBuffer
                      << " isPointDomain=" << computeProduct->isPointDomain
                      << " runtimeModelId=" << computeProduct->runtimeModelId << std::endl;
            return false;
        }

        outConfig = {};
        outConfig.showVoronoi = package.display.showVoronoi && package.domainType != DomainType::Points;
        outConfig.showPoints = package.display.showPoints;
        outConfig.nodeCount = computeProduct->nodeCount;
        outConfig.mappedVoronoiNodes = nullptr;
        outConfig.nodeBuffer = computeProduct->nodeBuffer;
        outConfig.nodeBufferOffset = computeProduct->nodeBufferOffset;
        outConfig.seedPositionBuffer = computeProduct->seedPositionBuffer;
        outConfig.seedPositionBufferOffset = computeProduct->seedPositionBufferOffset;
        outConfig.neighborIndicesBuffer = computeProduct->voronoiNeighborIndicesBuffer;
        outConfig.neighborIndicesBufferOffset = computeProduct->voronoiNeighborIndicesBufferOffset;
        outConfig.occupancyPointBuffer = computeProduct->occupancyPointBuffer;
        outConfig.occupancyPointBufferOffset = computeProduct->occupancyPointBufferOffset;
        outConfig.occupancyPointCount = computeProduct->occupancyPointCount;
        if (outConfig.showVoronoi) {
            const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.modelMeshHandle.key);
            const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelRemeshHandle.key);
            if (!modelProduct) {
                std::cerr << "[VoronoiDisplayTransport] missing ModelProduct for modelMeshHandle=" << package.modelMeshHandle.key << std::endl;
                return false;
            }
            if (!remeshProduct) {
                std::cerr << "[VoronoiDisplayTransport] missing RemeshProduct for modelRemeshHandle=" << package.modelRemeshHandle.key << std::endl;
                return false;
            }
            if (modelProduct->runtimeModelId != computeProduct->runtimeModelId) {
                std::cerr << "[VoronoiDisplayTransport] runtimeModelId mismatch: model=" << modelProduct->runtimeModelId
                          << " compute=" << computeProduct->runtimeModelId << std::endl;
                return false;
            }

            outConfig.bindingKey = socketKey;
            outConfig.runtimeModelId = computeProduct->runtimeModelId;
            outConfig.candidateBuffer = computeProduct->candidateBuffer;
            outConfig.candidateBufferOffset = computeProduct->candidateBufferOffset;
            outConfig.supportingHalfedgeView = remeshProduct->supportingHalfedgeView;
            outConfig.supportingAngleView = remeshProduct->supportingAngleView;
            outConfig.halfedgeView = remeshProduct->halfedgeView;
            outConfig.edgeView = remeshProduct->edgeView;
            outConfig.triangleView = remeshProduct->triangleView;
            outConfig.lengthView = remeshProduct->lengthView;
            outConfig.inputHalfedgeView = remeshProduct->inputHalfedgeView;
            outConfig.inputEdgeView = remeshProduct->inputEdgeView;
            outConfig.inputTriangleView = remeshProduct->inputTriangleView;
            outConfig.inputLengthView = remeshProduct->inputLengthView;
            outConfig.intrinsicVertexCount = remeshProduct->intrinsicVertexCount;
            outConfig.vertexBuffer = modelProduct->vertexBuffer;
            outConfig.vertexBufferOffset = modelProduct->vertexBufferOffset;
            outConfig.indexBuffer = modelProduct->indexBuffer;
            outConfig.indexBufferOffset = modelProduct->indexBufferOffset;
            outConfig.indexCount = modelProduct->indexCount;
            outConfig.modelMatrix = modelProduct->modelMatrix;
        }
        outConfig.displayHash = buildDisplayHash(outConfig, computeProduct->hashes.display);
        return true;
    }

    VoronoiDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
