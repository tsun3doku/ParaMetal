#pragma once

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/VoronoiDisplayController.hpp"

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
        }

        for (auto entity : registry.view<VoronoiPackage, Stale>()) {
            controller->remove(static_cast<uint64_t>(entity));
        }
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
        if (!computeProduct || !computeProduct->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.showVoronoi = package.display.showVoronoi;
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
        if (package.display.showVoronoi) {
            for (size_t i = 0; i < package.modelMeshHandles.size(); ++i) {
                if (i >= computeProduct->modelRuntimeModelIds.size()) {
                    break;
                }
                const uint64_t modelEntityId = package.modelMeshHandles[i].key;
                const uint64_t remeshEntityId = package.modelRemeshHandles[i].key;
                const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, modelEntityId);
                const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, remeshEntityId);
                if (!modelProduct || !remeshProduct) {
                    continue;
                }

                size_t voronoiIdx = 0;
                for (; voronoiIdx < computeProduct->modelRuntimeModelIds.size(); ++voronoiIdx) {
                    if (computeProduct->modelRuntimeModelIds[voronoiIdx] == modelProduct->runtimeModelId) {
                        break;
                    }
                }
                if (voronoiIdx >= computeProduct->modelRuntimeModelIds.size()) {
                    continue;
                }

                outConfig.modelRuntimeModelIds.push_back(modelProduct->runtimeModelId);
                outConfig.modelCandidateBuffers.push_back(computeProduct->modelCandidateBuffers[voronoiIdx]);
                outConfig.modelCandidateBufferOffsets.push_back(computeProduct->modelCandidateBufferOffsets[voronoiIdx]);
                outConfig.modelGMLSSurfaceStencilBuffers.push_back(computeProduct->modelGMLSSurfaceStencilBuffers[voronoiIdx]);
                outConfig.modelGMLSSurfaceStencilBufferOffsets.push_back(computeProduct->modelGMLSSurfaceStencilBufferOffsets[voronoiIdx]);
                outConfig.modelGMLSSurfaceWeightBuffers.push_back(computeProduct->modelGMLSSurfaceWeightBuffers[voronoiIdx]);
                outConfig.modelGMLSSurfaceWeightBufferOffsets.push_back(computeProduct->modelGMLSSurfaceWeightBufferOffsets[voronoiIdx]);
                outConfig.modelGMLSSurfaceGradientWeightBuffers.push_back(computeProduct->modelGMLSSurfaceGradientWeightBuffers[voronoiIdx]);
                outConfig.modelGMLSSurfaceGradientWeightBufferOffsets.push_back(computeProduct->modelGMLSSurfaceGradientWeightBufferOffsets[voronoiIdx]);
                outConfig.modelSupportingHalfedgeViews.push_back(remeshProduct->supportingHalfedgeView);
                outConfig.modelSupportingAngleViews.push_back(remeshProduct->supportingAngleView);
                outConfig.modelHalfedgeViews.push_back(remeshProduct->halfedgeView);
                outConfig.modelEdgeViews.push_back(remeshProduct->edgeView);
                outConfig.modelTriangleViews.push_back(remeshProduct->triangleView);
                outConfig.modelLengthViews.push_back(remeshProduct->lengthView);
                outConfig.modelInputHalfedgeViews.push_back(remeshProduct->inputHalfedgeView);
                outConfig.modelInputEdgeViews.push_back(remeshProduct->inputEdgeView);
                outConfig.modelInputTriangleViews.push_back(remeshProduct->inputTriangleView);
                outConfig.modelInputLengthViews.push_back(remeshProduct->inputLengthView);
                outConfig.modelIntrinsicVertexCounts.push_back(remeshProduct->intrinsicVertexCount);
                outConfig.modelVertexBuffers.push_back(modelProduct->vertexBuffer);
                outConfig.modelVertexBufferOffsets.push_back(modelProduct->vertexBufferOffset);
                outConfig.modelIndexBuffers.push_back(modelProduct->indexBuffer);
                outConfig.modelIndexBufferOffsets.push_back(modelProduct->indexBufferOffset);
                outConfig.modelIndexCounts.push_back(modelProduct->indexCount);
                outConfig.modelMatrices.push_back(modelProduct->modelMatrix);
            }
        }
        outConfig.displayHash = buildDisplayHash(outConfig, computeProduct->productHash);
        return true;
    }

    VoronoiDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
};
