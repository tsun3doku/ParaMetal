#include "RuntimeModelComputeTransport.hpp"

#include "nodegraph/NodeModelTransform.hpp"
#include "runtime/ModelComputeRuntime.hpp"

#include <iostream>

void RuntimeModelComputeTransport::sync(const ECSRegistry& registry) {
    if (!modelRuntime) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;

    auto view = registry.view<ModelPackage>(entt::exclude<Stale>);
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<ModelPackage>(entity);
        nextSocketKeys.insert(socketKey);

        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        const ModelProduct* product = tryGetProduct<ModelProduct>(registry, socketKey);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        if (inputHash != 0 && product && hashIt != appliedConfigInputHash.end() && hashIt->second == inputHash) {
            continue;
        }

        const std::string* modelPath = nullptr;
        if (!tryBuildRuntimeModelPath(socketKey, package, modelPath)) {
            continue;
        }

        modelRuntime->queueAcquireSocket(socketKey, *modelPath);
        nextSocketKeys.insert(socketKey);
    }

    for (auto entity : registry.view<ModelPackage, Stale>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        modelRuntime->queueReleaseSocket(socketKey);
        appliedConfigInputHash.erase(socketKey);
    }

    activeSocketKeys = std::move(nextSocketKeys);
}

void RuntimeModelComputeTransport::finalizeSync() {
    if (!modelRuntime) {
        return;
    }

    modelRuntime->flush();

    auto staleProductView = ecsRegistry->view<ModelProduct, Stale>();
    for (auto entity : staleProductView) {
        removePublishedProduct(static_cast<uint64_t>(entity));
    }
    for (uint64_t socketKey : activeSocketKeys) {
        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<ModelPackage>(entity);
        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        const ModelProduct* product = tryGetProduct<ModelProduct>(*ecsRegistry, socketKey);
        if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash) {
            uint32_t runtimeModelId = 0;
            if (modelRuntime->tryGetRuntimeModelId(socketKey, runtimeModelId) && runtimeModelId != 0) {
                publishProduct(socketKey, runtimeModelId);
            } else {
                std::cerr << "[ModelCompute] No runtime model ID for socketKey=" << socketKey
                          << ", baseModelPath='" << package.geometry.baseModelPath << "'" << std::endl;
                removePublishedProduct(socketKey);
            }
        }
    }
}

bool RuntimeModelComputeTransport::tryBuildRuntimeModelPath(
    uint64_t socketKey,
    const ModelPackage& package,
    const std::string*& outModelPath) const {
    outModelPath = nullptr;
    if (socketKey == 0 || package.geometry.baseModelPath.empty()) {
        return false;
    }

    outModelPath = &package.geometry.baseModelPath;
    return true;
}

uint64_t RuntimeModelComputeTransport::buildConfigInputHash(uint64_t socketKey, const ModelPackage& package) const {
    (void)socketKey;
    return package.packageHash;
}

void RuntimeModelComputeTransport::removePublishedProduct(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);
    ecsRegistry->remove<ModelProduct>(entity);
}

void RuntimeModelComputeTransport::publishProduct(uint64_t socketKey, uint32_t runtimeModelId) {
    if (!modelRuntime || socketKey == 0 || runtimeModelId == 0) {
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);
    const auto& package = ecsRegistry->get<ModelPackage>(entity);
    modelRuntime->setModelMatrix(runtimeModelId, NodeModelTransform::toMat4(package.localToWorld));

    ModelProduct product{};
    if (!modelRuntime->exportProduct(runtimeModelId, product)) {
        ecsRegistry->remove<ModelProduct>(entity);
        return;
    }

    ecsRegistry->emplace_or_replace<ModelProduct>(entity, product);
    appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
}
