#pragma once

#include "runtime/RemeshController.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"

#include <unordered_set>
#include <vector>

class RuntimeRemeshComputeTransport {
public:
    void setController(RemeshController* updatedController) {
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

        auto view = registry.view<RemeshPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            const auto& package = registry.get<RemeshPackage>(entity);

            const uint64_t inputHash = buildConfigInputHash(socketKey, package);
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(registry, socketKey);
            auto hashIt = appliedConfigInputHash.find(socketKey);
            if (inputHash != 0 && product && hashIt != appliedConfigInputHash.end() && hashIt->second == inputHash) {
                nextSocketKeys.insert(socketKey);
                continue;
            }

            RemeshController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                continue;
            }

            controller->configure(config);
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
            const auto& package = ecsRegistry->get<RemeshPackage>(entity);
            const uint64_t inputHash = buildConfigInputHash(socketKey, package);
            auto hashIt = appliedConfigInputHash.find(socketKey);
            const RemeshProduct* product = tryGetProduct<RemeshProduct>(*ecsRegistry, socketKey);
            if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash) {
                publishProduct(socketKey);
            }
        }
    }

private:
    bool tryBuildConfig(uint64_t socketKey, const RemeshPackage& package, RemeshController::Config& outConfig) const {
        if (socketKey == 0 || !ecsRegistry) {
            return false;
        }

        outConfig = {};
        outConfig.socketKey = socketKey;
        outConfig.pointPositions = package.sourceGeometry.pointPositions;
        outConfig.triangleIndices = package.sourceGeometry.triangleIndices;
        outConfig.iterations = package.iterations;
        outConfig.minAngleDegrees = package.minAngleDegrees;
        outConfig.maxEdgeLength = package.maxEdgeLength;
        outConfig.stepSize = package.stepSize;

        const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.sourceMeshHandle.key);
        outConfig.runtimeModelId = modelProduct ? modelProduct->runtimeModelId : 0;
        outConfig.computeHash = buildConfigInputHash(socketKey, package);
        return true;
    }

    uint64_t buildConfigInputHash(uint64_t socketKey, const RemeshPackage& package) const {
        (void)socketKey;
        uint64_t hash = package.packageHash;
        const ModelProduct* modelProduct = tryGetProduct<ModelProduct>(*ecsRegistry, package.sourceMeshHandle.key);
        if (!modelProduct) {
            return 0;
        }
        NodeGraphHash::combine(hash, modelProduct->productHash);
        return hash;
    }

    void removePublishedProduct(uint64_t socketKey) {
        if (socketKey == 0) {
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        ecsRegistry->remove<RemeshProduct>(entity);
    }

    void publishProduct(uint64_t socketKey) {
        if (!controller || socketKey == 0) {
            return;
        }

        RemeshProduct product{};
        if (!buildProduct(socketKey, product)) {
            auto entity = static_cast<ECSEntity>(socketKey);
            ecsRegistry->remove<RemeshProduct>(entity);
            return;
        }

        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<RemeshPackage>(entity);
        ecsRegistry->emplace_or_replace<RemeshProduct>(entity, product);
        appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
    }

    bool buildProduct(uint64_t socketKey, RemeshProduct& outProduct) const {
        outProduct = {};
        const RemeshSystem* system = controller ? controller->getSystem(socketKey) : nullptr;
        if (!system || !system->isReady()) {
            return false;
        }

        const SupportingHalfedge::GPUResources& gpu = system->getIntrinsicGpuResources();
        outProduct.runtimeModelId = system->getRuntimeModelId();
        outProduct.geometryPositions = system->getGeometryPositions();
        outProduct.geometryTriangleIndices = system->getGeometryTriangleIndices();
        outProduct.intrinsicMesh = system->getIntrinsicMesh();
        outProduct.intrinsicTriangleBuffer = gpu.intrinsicTriangleBuffer;
        outProduct.intrinsicTriangleBufferOffset = gpu.triangleGeometryOffset;
        outProduct.intrinsicVertexBuffer = gpu.intrinsicVertexBuffer;
        outProduct.intrinsicVertexBufferOffset = gpu.vertexGeometryOffset;
        outProduct.intrinsicTriangleCount = gpu.triangleCount;
        outProduct.intrinsicVertexCount = gpu.vertexCount;
        outProduct.averageTriangleArea = gpu.averageTriangleArea;
        outProduct.supportingHalfedgeView = gpu.viewS;
        outProduct.supportingAngleView = gpu.viewA;
        outProduct.halfedgeView = gpu.viewH;
        outProduct.edgeView = gpu.viewE;
        outProduct.triangleView = gpu.viewT;
        outProduct.lengthView = gpu.viewL;
        outProduct.inputHalfedgeView = gpu.viewHInput;
        outProduct.inputEdgeView = gpu.viewEInput;
        outProduct.inputTriangleView = gpu.viewTInput;
        outProduct.inputLengthView = gpu.viewLInput;
        outProduct.productHash = buildProductHash(outProduct);
        return outProduct.isValid();
    }

    RemeshController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    std::unordered_map<uint64_t, uint64_t> appliedConfigInputHash;
};
