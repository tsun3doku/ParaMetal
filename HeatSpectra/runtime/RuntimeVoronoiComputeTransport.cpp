#include "RuntimeVoronoiComputeTransport.hpp"
#include "heat/VoronoiSystemComputeController.hpp"


void RuntimeVoronoiComputeTransport::sync(const ECSRegistry& registry) {
    if (!controller) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;

    auto view = registry.view<VoronoiPackage>(entt::exclude<Stale>);
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<VoronoiPackage>(entity);

        if (!package.authored.active || package.modelMeshHandles.empty()) {
            controller->disable(socketKey);
            appliedConfigInputHash.erase(socketKey);
            continue;
        }
        nextSocketKeys.insert(socketKey);

        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(registry, socketKey);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        if (inputHash != 0 && product && hashIt != appliedConfigInputHash.end() && hashIt->second == inputHash) {
            continue;
        }

        VoronoiSystemComputeController::Config config{};
        if (!tryBuildConfig(socketKey, package, config)) {
            continue;
        }

        controller->configure(socketKey, config);
    }

    for (auto entity : registry.view<VoronoiPackage, Stale>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        controller->disable(socketKey);
        appliedConfigInputHash.erase(socketKey);
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
    if (!package.authored.active || package.modelMeshHandles.empty()) {
        return false;
    }

    outConfig = {};
    outConfig.active = true;
    outConfig.cellSize = package.authored.cellSize;
    outConfig.voxelResolution = package.authored.voxelResolution;

    for (size_t modelIndex = 0; modelIndex < package.modelMeshHandles.size(); ++modelIndex) {
        if (modelIndex >= package.modelRemeshHandles.size()) {
            return false;
        }

        const uint64_t remeshEntityId = package.modelRemeshHandles[modelIndex].key;
        const RemeshProduct* remeshProduct =
            tryGetProduct<RemeshProduct>(*ecsRegistry, remeshEntityId);
        if (!remeshProduct) {
            return false;
        }

        const uint64_t modelEntityId = package.modelMeshHandles[modelIndex].key;
        const ModelProduct* modelProduct =
            tryGetProduct<ModelProduct>(*ecsRegistry, modelEntityId);
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
        outConfig.meshModelMatrices.push_back(NodeModelTransform::toMat4(package.modelLocalToWorlds[modelIndex]));
    }

    outConfig.computeHash = buildComputeHash(outConfig);
    return true;
}

void RuntimeVoronoiComputeTransport::finalizeSync() {
    if (!controller) {
        return;
    }

    auto staleProductView = ecsRegistry->view<VoronoiProduct, Stale>();
    for (auto entity : staleProductView) {
        removePublishedProduct(static_cast<uint64_t>(entity));
    }
    for (uint64_t socketKey : activeSocketKeys) {
        auto entity = static_cast<ECSEntity>(socketKey);
        if (!ecsRegistry->valid(entity) || !ecsRegistry->all_of<VoronoiPackage>(entity)) {
            continue;
        }
        const auto& package = ecsRegistry->get<VoronoiPackage>(entity);
        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(*ecsRegistry, socketKey);
        if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash) {
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
    if (!buildProduct(socketKey, product)) {
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
    appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
}

uint64_t RuntimeVoronoiComputeTransport::buildConfigInputHash(uint64_t socketKey, const VoronoiPackage& package) const {
    (void)socketKey;
    uint64_t hash = package.packageHash;
    for (size_t modelIndex = 0; modelIndex < package.modelMeshHandles.size(); ++modelIndex) {
        if (modelIndex >= package.modelRemeshHandles.size()) {
            return 0;
        }
        const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelRemeshHandles[modelIndex].key);
        const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.modelMeshHandles[modelIndex].key);
        if (!remeshProduct || !modelProduct) {
            return 0;
        }
        NodeGraphHash::combine(hash, remeshProduct->productHash);
        NodeGraphHash::combine(hash, modelProduct->productHash);
    }
    return hash;
}

bool RuntimeVoronoiComputeTransport::buildProduct(uint64_t socketKey, VoronoiProduct& outProduct) const {
    outProduct = {};
    const VoronoiSystem* voronoiSystem = controller ? controller->getSystem(socketKey) : nullptr;
    if (!voronoiSystem || !controller->getConfig(socketKey)) {
        return false;
    }

    outProduct.nodeCount = voronoiSystem->getVoronoiNodeCount();
    outProduct.simNodeCount = voronoiSystem->runtimeRef().getSimNodeCount();

    const VoronoiResources& resources = voronoiSystem->resourcesRef();
    outProduct.mappedVoronoiNodes = nullptr;
    outProduct.nodeBuffer = resources.voronoiNodeBuffer;
    outProduct.nodeBufferOffset = resources.voronoiNodeBufferOffset;
    outProduct.voronoiNeighborBuffer = resources.voronoiNeighborBuffer;
    outProduct.voronoiNeighborBufferOffset = resources.voronoiNeighborBufferOffset;
    outProduct.voronoiNeighborIndicesBuffer = resources.voronoiNeighborIndicesBuffer;
    outProduct.voronoiNeighborIndicesBufferOffset = resources.voronoiNeighborIndicesBufferOffset;
    outProduct.voronoiInterfaceAreasBuffer = resources.voronoiInterfaceAreasBuffer;
    outProduct.voronoiInterfaceAreasBufferOffset = resources.voronoiInterfaceAreasBufferOffset;
    outProduct.voronoiInterfaceNeighborIdsBuffer = resources.voronoiInterfaceNeighborIdsBuffer;
    outProduct.voronoiInterfaceNeighborIdsBufferOffset = resources.voronoiInterfaceNeighborIdsBufferOffset;
    outProduct.voronoiGMLSInterfaceBuffer = resources.voronoiGMLSInterfaceBuffer;
    outProduct.voronoiGMLSInterfaceBufferOffset = resources.voronoiGMLSInterfaceBufferOffset;
    outProduct.simNodeBuffer = voronoiSystem->runtimeRef().getSimNodeBuffer();
    outProduct.simNodeBufferOffset = voronoiSystem->runtimeRef().getSimNodeBufferOffset();
    outProduct.simGMLSInterfaceBuffer = voronoiSystem->runtimeRef().getSimGMLSInterfaceBuffer();
    outProduct.simGMLSInterfaceBufferOffset = voronoiSystem->runtimeRef().getSimGMLSInterfaceBufferOffset();
    outProduct.simGMLSInterfaceCount = voronoiSystem->runtimeRef().getSimGMLSInterfaceCount();
    outProduct.voronoiSeedFlagsBuffer = resources.voronoiSeedFlagsBuffer;
    outProduct.voronoiSeedFlagsBufferOffset = resources.voronoiSeedFlagsBufferOffset;
    outProduct.seedPositionBuffer = resources.seedPositionBuffer;
    outProduct.seedPositionBufferOffset = resources.seedPositionBufferOffset;
    outProduct.occupancyPointBuffer = resources.occupancyPointBuffer;
    outProduct.occupancyPointBufferOffset = resources.occupancyPointBufferOffset;
    outProduct.occupancyPointCount = resources.occupancyPointCount;
    outProduct.voronoiToSim = voronoiSystem->runtimeRef().getVoronoiToSim();
    outProduct.simToVoronoi = voronoiSystem->runtimeRef().getSimToVoronoi();

    const auto& modelRuntimes = voronoiSystem->getModelRuntimes();
    const auto& domainSeedFlags = voronoiSystem->runtimeRef().getSeedFlags();
    const auto& domainSeedPositions = voronoiSystem->runtimeRef().getSeedPositions();

    outProduct.modelRuntimeModelIds.reserve(modelRuntimes.size());
    outProduct.modelCandidateBuffers.reserve(modelRuntimes.size());
    outProduct.modelCandidateBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceStencilBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceStencilBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceWeightBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceWeightBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceGradientWeightBuffers.reserve(modelRuntimes.size());
    outProduct.modelGMLSSurfaceGradientWeightBufferOffsets.reserve(modelRuntimes.size());
    outProduct.modelSeedFlags.reserve(modelRuntimes.size());
    outProduct.modelSeedPositions.reserve(modelRuntimes.size());

    for (const auto& modelRuntime : modelRuntimes) {
        if (!modelRuntime) {
            continue;
        }
        outProduct.modelRuntimeModelIds.push_back(modelRuntime->getRuntimeModelId());
        outProduct.modelCandidateBuffers.push_back(modelRuntime->getVoronoiCandidateBuffer());
        outProduct.modelCandidateBufferOffsets.push_back(modelRuntime->getVoronoiCandidateBufferOffset());
        outProduct.modelGMLSSurfaceStencilBuffers.push_back(modelRuntime->getGMLSSurfaceStencilBuffer());
        outProduct.modelGMLSSurfaceStencilBufferOffsets.push_back(modelRuntime->getGMLSSurfaceStencilBufferOffset());
        outProduct.modelGMLSSurfaceWeightBuffers.push_back(modelRuntime->getGMLSSurfaceWeightBuffer());
        outProduct.modelGMLSSurfaceWeightBufferOffsets.push_back(modelRuntime->getGMLSSurfaceWeightBufferOffset());
        outProduct.modelGMLSSurfaceWeightCounts.push_back(modelRuntime->getGMLSSurfaceWeightCount());
        outProduct.modelGMLSSurfaceGradientWeightBuffers.push_back(modelRuntime->getGMLSSurfaceGradientWeightBuffer());
        outProduct.modelGMLSSurfaceGradientWeightBufferOffsets.push_back(modelRuntime->getGMLSSurfaceGradientWeightBufferOffset());
        outProduct.modelGMLSSurfaceGradientWeightCounts.push_back(modelRuntime->getGMLSSurfaceGradientWeightCount());
        outProduct.modelSeedFlags.push_back(domainSeedFlags);
        std::vector<glm::vec3> seedPositions;
        seedPositions.reserve(domainSeedPositions.size());
        for (const glm::vec4& pos : domainSeedPositions) {
            seedPositions.push_back(glm::vec3(pos));
        }
        outProduct.modelSeedPositions.push_back(std::move(seedPositions));
    }

    outProduct.productHash = buildProductHash(outProduct);
    return outProduct.isValid();
}
