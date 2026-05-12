#include "RuntimeVoronoiComputeTransport.hpp"
#include "heat/VoronoiSystemComputeController.hpp"


void RuntimeVoronoiComputeTransport::sync(const ECSRegistry& registry) {
    if (!controller) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;

    auto view = registry.view<VoronoiPackage>();
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<VoronoiPackage>(entity);

        if (!package.authored.active || package.modelProducts.empty()) {
            continue;
        }
        nextSocketKeys.insert(socketKey);

        const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(registry, socketKey);
        auto hashIt = appliedPackageHash.find(socketKey);
        if (product && hashIt != appliedPackageHash.end() && hashIt->second == package.packageHash) {
            continue;
        }

        VoronoiSystemComputeController::Config config{};
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

bool RuntimeVoronoiComputeTransport::tryBuildConfig(
    uint64_t socketKey,
    const VoronoiPackage& package,
    VoronoiSystemComputeController::Config& outConfig) const {
    if (socketKey == 0 || !ecsRegistry) {
        return false;
    }
    if (!package.authored.active || package.modelProducts.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.active = true;
    outConfig.cellSize = package.authored.cellSize;
    outConfig.voxelResolution = package.authored.voxelResolution;

    for (size_t modelIndex = 0; modelIndex < package.modelProducts.size(); ++modelIndex) {
        if (modelIndex >= package.modelRemeshProducts.size()) {
            return false;
        }

        const ProductHandle& remeshProductHandle = package.modelRemeshProducts[modelIndex];
        const RemeshProduct* remeshProduct =
            tryGetProduct<RemeshProduct>(*ecsRegistry, remeshProductHandle.outputSocketKey);
        if (!remeshProduct) {
            return false;
        }

        const ProductHandle& modelProductHandle = package.modelProducts[modelIndex];
        const ModelProduct* modelProduct =
            tryGetProduct<ModelProduct>(*ecsRegistry, modelProductHandle.outputSocketKey);
        if (!modelProduct) {
            return false;
        }

        outConfig.receiverRuntimeModelIds.push_back(remeshProduct->runtimeModelId);
        outConfig.receiverNodeModelIds.push_back(0);
        outConfig.receiverGeometryPositions.push_back(remeshProduct->geometryPositions);
        outConfig.receiverGeometryTriangleIndices.push_back(remeshProduct->geometryTriangleIndices);
        outConfig.receiverIntrinsicMeshes.push_back(remeshProduct->intrinsicMesh);
        outConfig.receiverIntrinsicTriangleIndices.push_back(remeshProduct->intrinsicMesh.indices);
        outConfig.receiverSurfaceVertices.emplace_back().reserve(remeshProduct->intrinsicMesh.vertices.size());
        for (const SupportingHalfedge::IntrinsicVertex& intrinsicVertex : remeshProduct->intrinsicMesh.vertices) {
            VoronoiModelRuntime::SurfaceVertex vertex{};
            vertex.position = glm::vec4(intrinsicVertex.position, 1.0f);
            vertex.normal = glm::vec4(intrinsicVertex.normal, 0.0f);
            outConfig.receiverSurfaceVertices.back().push_back(vertex);
        }
        outConfig.supportingHalfedgeViews.push_back(remeshProduct->supportingHalfedgeView);
        outConfig.supportingAngleViews.push_back(remeshProduct->supportingAngleView);
        outConfig.halfedgeViews.push_back(remeshProduct->halfedgeView);
        outConfig.edgeViews.push_back(remeshProduct->edgeView);
        outConfig.triangleViews.push_back(remeshProduct->triangleView);
        outConfig.lengthViews.push_back(remeshProduct->lengthView);
        outConfig.inputHalfedgeViews.push_back(remeshProduct->inputHalfedgeView);
        outConfig.inputEdgeViews.push_back(remeshProduct->inputEdgeView);
        outConfig.inputTriangleViews.push_back(remeshProduct->inputTriangleView);
        outConfig.inputLengthViews.push_back(remeshProduct->inputLengthView);

        outConfig.meshVertexBuffers.push_back(modelProduct->vertexBuffer);
        outConfig.meshVertexBufferOffsets.push_back(modelProduct->vertexBufferOffset);
        outConfig.meshIndexBuffers.push_back(modelProduct->indexBuffer);
        outConfig.meshIndexBufferOffsets.push_back(modelProduct->indexBufferOffset);
        outConfig.meshIndexCounts.push_back(modelProduct->indexCount);
        outConfig.meshModelMatrices.push_back(NodeModelTransform::toMat4(package.modelLocalToWorlds[modelIndex]));
    }

    outConfig.computeHash = buildComputeHash(outConfig);
    return true;
}

void RuntimeVoronoiComputeTransport::finalizeSync() {
    if (!controller) {
        return;
    }

    std::vector<uint64_t> removals;
    auto productView = ecsRegistry->view<VoronoiProduct>();
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
        if (!ecsRegistry->valid(entity) || !ecsRegistry->all_of<VoronoiPackage>(entity)) {
            continue;
        }
        const auto& package = ecsRegistry->get<VoronoiPackage>(entity);
        auto hashIt = appliedPackageHash.find(socketKey);
        const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(*ecsRegistry, socketKey);
        if (!product || hashIt == appliedPackageHash.end() || hashIt->second != package.packageHash) {
            publishProduct(socketKey);
        }
    }
}

void RuntimeVoronoiComputeTransport::removePublishedProduct(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);
    ecsRegistry->remove<VoronoiProduct>(entity);
}

void RuntimeVoronoiComputeTransport::publishProduct(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }

    VoronoiProduct product{};
    if (!controller->exportProduct(socketKey, product)) {
        auto entity = static_cast<ECSEntity>(socketKey);
        ecsRegistry->remove<VoronoiProduct>(entity);
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);
    if (!ecsRegistry->valid(entity) || !ecsRegistry->all_of<VoronoiPackage>(entity)) {
        return;
    }
    const auto& package = ecsRegistry->get<VoronoiPackage>(entity);

    ecsRegistry->emplace_or_replace<VoronoiProduct>(entity, product);
    appliedPackageHash[socketKey] = package.packageHash;
}
