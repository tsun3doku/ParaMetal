#include "RuntimeVoronoiComputeTransport.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProduct.hpp"
#include "heat/VoronoiSystemComputeController.hpp"
#include "runtime/RuntimePackages.hpp"
#include "util/GeometryUtils.hpp"

void RuntimeVoronoiComputeTransport::sync(const ECSRegistry& registry) {
    if (!controller) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;

    auto view = registry.view<VoronoiPackage>(entt::exclude<Stale>);
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<VoronoiPackage>(entity);

        bool inputsReady = false;
        if (package.domainType == DomainType::Mesh) {
            inputsReady = package.modelMeshHandle.key != 0 && package.modelRemeshHandle.key != 0 && package.pointsPayloadHandle.key != 0;
        } else {
            inputsReady = package.pointsPayloadHandle.key != 0;
        }
        if (!package.authored.active || !inputsReady) {
            controller->disable(socketKey);
            removePublishedProduct(socketKey);
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

    for (uint64_t socketKey : activeSocketKeys) {
        if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
            controller->disable(socketKey);
            removePublishedProduct(socketKey);
            appliedConfigInputHash.erase(socketKey);
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

    outConfig = {};
    outConfig.active = true;
    outConfig.cellSize = package.authored.cellSize;
    outConfig.voxelResolution = package.authored.voxelResolution;

    if (package.domainType == DomainType::Mesh) {
        if (!package.authored.active || package.modelMeshHandle.key == 0 || package.modelRemeshHandle.key == 0) {
            return false;
        }

        const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelRemeshHandle.key);
        const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.modelMeshHandle.key);
        if (!remeshProduct || !modelProduct || remeshProduct->runtimeModelId == 0) {
            return false;
        }

        outConfig.receiverRuntimeModelId = remeshProduct->runtimeModelId;
        outConfig.receiverNodeModelId = 0;
        outConfig.receiverGeometryPositions = remeshProduct->geometryPositions;
        outConfig.receiverGeometryTriangleIndices = remeshProduct->geometryTriangleIndices;
        outConfig.receiverIntrinsicMesh = remeshProduct->intrinsicMesh;
        outConfig.receiverIntrinsicTriangleIndices = remeshProduct->intrinsicMesh.indices;
        outConfig.receiverSurfaceVertices.reserve(remeshProduct->intrinsicMesh.vertices.size());
        for (const SupportingHalfedge::IntrinsicVertex& intrinsicVertex : remeshProduct->intrinsicMesh.vertices) {
            VoronoiModelRuntime::SurfaceVertex vertex{};
            vertex.position = glm::vec4(intrinsicVertex.position, 1.0f);
            vertex.normal = glm::vec4(intrinsicVertex.normal, 0.0f);
            outConfig.receiverSurfaceVertices.push_back(vertex);
        }
        outConfig.meshModelMatrix = toMat4(package.modelLocalToWorld);

        if (package.pointsPayloadHandle.key != 0 && ecsRegistry) {
            auto pointsEntity = static_cast<ECSEntity>(package.pointsPayloadHandle.key);
            if (ecsRegistry->valid(pointsEntity) && ecsRegistry->all_of<PointPackage>(pointsEntity)) {
                const auto& pointPackage = ecsRegistry->get<PointPackage>(pointsEntity);
                if (!pointPackage.positions.empty()) {
                    outConfig.pointPositions = pointPackage.positions;
                    const glm::mat4 pointsToMeshLocal = glm::inverse(toMat4(package.modelLocalToWorld)) * toMat4(pointPackage.localToWorld);
                    for (glm::vec4& pos : outConfig.pointPositions) {
                        pos = pointsToMeshLocal * pos;
                    }
                }
            }
        }
    } else if (package.domainType == DomainType::Points) {
        if (!package.authored.active || package.pointsPayloadHandle.key == 0 || !ecsRegistry) {
            return false;
        }

        auto pointsEntity = static_cast<ECSEntity>(package.pointsPayloadHandle.key);
        if (!ecsRegistry->valid(pointsEntity) || !ecsRegistry->all_of<PointPackage>(pointsEntity)) {
            return false;
        }

        const auto& pointPackage = ecsRegistry->get<PointPackage>(pointsEntity);
        if (pointPackage.positions.empty()) {
            return false;
        }

        outConfig.isPointDomain = true;
        outConfig.pointPositions = pointPackage.positions;
    }

    outConfig.computeHash = buildConfigInputHash(socketKey, package);
    return true;
}

void RuntimeVoronoiComputeTransport::finalizeSync() {
    if (!controller) {
        return;
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
    uint64_t hash = package.hashes.simulation;

    if (package.pointsPayloadHandle.key != 0 && ecsRegistry) {
        auto pointsEntity = static_cast<ECSEntity>(package.pointsPayloadHandle.key);
        if (ecsRegistry->valid(pointsEntity) && ecsRegistry->all_of<PointPackage>(pointsEntity)) {
            const auto& pointPackage = ecsRegistry->get<PointPackage>(pointsEntity);
            HashBuilder::combine(hash, pointPackage.hashes.geometry);
        }
    }

    if (package.domainType == DomainType::Mesh) {
        const RemeshProduct* remeshProduct = tryGetProduct<RemeshProduct>(*ecsRegistry, package.modelRemeshHandle.key);
        const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.modelMeshHandle.key);
        if (!remeshProduct || !modelProduct) {
            return 0;
        }
        HashBuilder::combine(hash, remeshProduct->hashes.geometry);
        HashBuilder::combine(hash, modelProduct->hashes.geometry);
    }
    return hash;
}

bool RuntimeVoronoiComputeTransport::buildProduct(uint64_t socketKey, VoronoiProduct& outProduct) const {
    outProduct = {};
    const VoronoiSystem* voronoiSystem = controller ? controller->getSystem(socketKey) : nullptr;
    if (!voronoiSystem) {
        return false;
    }
    if (!controller->getConfig(socketKey)) {
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

    const VoronoiDomainRuntime* domainRuntime = voronoiSystem->getDomainRuntime();
    if (!domainRuntime) {
        return false;
    }
    const auto& domainSeedFlags = voronoiSystem->runtimeRef().getSeedFlags();
    const auto& domainSeedPositions = voronoiSystem->runtimeRef().getSeedPositions();

    outProduct.isPointDomain = domainRuntime->isPointDomain();
    outProduct.candidateBuffer = domainRuntime->getCandidateBuffer();
    outProduct.candidateBufferOffset = domainRuntime->getCandidateBufferOffset();

    if (!domainRuntime->isPointDomain()) {
        const VoronoiModelRuntime* modelRuntime = static_cast<const VoronoiModelRuntime*>(domainRuntime);
        outProduct.runtimeModelId = modelRuntime->getRuntimeModelId();
        outProduct.gmlsSurfaceStencilBuffer = modelRuntime->getGMLSSurfaceStencilBuffer();
        outProduct.gmlsSurfaceStencilBufferOffset = modelRuntime->getGMLSSurfaceStencilBufferOffset();
        outProduct.gmlsSurfaceWeightBuffer = modelRuntime->getGMLSSurfaceWeightBuffer();
        outProduct.gmlsSurfaceWeightBufferOffset = modelRuntime->getGMLSSurfaceWeightBufferOffset();
        outProduct.gmlsSurfaceWeightCount = modelRuntime->getGMLSSurfaceWeightCount();
        outProduct.gmlsSurfaceGradientWeightBuffer = modelRuntime->getGMLSSurfaceGradientWeightBuffer();
        outProduct.gmlsSurfaceGradientWeightBufferOffset = modelRuntime->getGMLSSurfaceGradientWeightBufferOffset();
        outProduct.gmlsSurfaceGradientWeightCount = modelRuntime->getGMLSSurfaceGradientWeightCount();
    } else {
        outProduct.runtimeModelId = 0;
    }

    outProduct.seedFlags = domainSeedFlags;
    outProduct.seedPositions.reserve(domainSeedPositions.size());
    for (const glm::vec4& pos : domainSeedPositions) {
        outProduct.seedPositions.push_back(glm::vec3(pos));
    }

    HashProduct::seal(outProduct);
    if (!outProduct.isValid()) {
        return false;
    }
    return true;
}
