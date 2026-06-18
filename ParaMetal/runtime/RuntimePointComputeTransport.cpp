#include "RuntimePointComputeTransport.hpp"

#include "runtime/PointComputeRuntime.hpp"
#include "hash/HashBuilder.hpp"
#include "util/GeometryUtils.hpp"

#include <iostream>

void RuntimePointComputeTransport::sync(const ECSRegistry& registry) {
    if (!runtime) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;

    auto view = registry.view<PointPackage>(entt::exclude<Stale>);
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        const auto& package = registry.get<PointPackage>(entity);

        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        if (inputHash != 0 && hashIt != appliedConfigInputHash.end() && hashIt->second == inputHash) {
            nextSocketKeys.insert(socketKey);
            continue;
        }

        if (package.pointCount == 0 || package.positions.empty()) {
            runtime->disable(socketKey);
            removePublishedProduct(socketKey);
            appliedConfigInputHash.erase(socketKey);
            continue;
        }

        PointComputeRuntime::Config config{};
        if (!tryBuildConfig(socketKey, package, config)) {
            continue;
        }
        runtime->configure(config);
        nextSocketKeys.insert(socketKey);
    }

    for (uint64_t socketKey : activeSocketKeys) {
        if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
            runtime->disable(socketKey);
            removePublishedProduct(socketKey);
            appliedConfigInputHash.erase(socketKey);
        }
    }

    activeSocketKeys = std::move(nextSocketKeys);
}

void RuntimePointComputeTransport::finalizeSync() {
    if (!runtime || !ecsRegistry) {
        return;
    }

    for (uint64_t socketKey : activeSocketKeys) {
        auto entity = static_cast<ECSEntity>(socketKey);
        const auto& package = ecsRegistry->get<PointPackage>(entity);
        const uint64_t inputHash = buildConfigInputHash(socketKey, package);
        auto hashIt = appliedConfigInputHash.find(socketKey);
        const PointProduct* product = tryGetProduct<PointProduct>(*ecsRegistry, socketKey);
        if (!product || hashIt == appliedConfigInputHash.end() || hashIt->second != inputHash) {
            publishProduct(socketKey);
        }
    }
}

bool RuntimePointComputeTransport::tryBuildConfig(
    uint64_t socketKey,
    const PointPackage& package,
    PointComputeRuntime::Config& outConfig) const {
    outConfig = {};
    if (socketKey == 0 || package.pointCount == 0 || package.positions.empty()) {
        return false;
    }
    outConfig.socketKey = socketKey;
    outConfig.positions = package.positions;
    outConfig.modelMatrix = toMat4(package.localToWorld);
    outConfig.computeHash = buildConfigInputHash(socketKey, package);
    return true;
}

uint64_t RuntimePointComputeTransport::buildConfigInputHash(uint64_t socketKey, const PointPackage& package) const {
    (void)socketKey;
    uint64_t hash = package.hashes.geometry;
    const PointProduct* product = tryGetProduct<PointProduct>(*ecsRegistry, socketKey);
    if (product) {
        HashBuilder::combine(hash, product->hashes.geometry);
    }
    return hash;
}

void RuntimePointComputeTransport::removePublishedProduct(uint64_t socketKey) {
    if (socketKey == 0 || !ecsRegistry) {
        return;
    }
    auto entity = static_cast<ECSEntity>(socketKey);
    ecsRegistry->remove<PointProduct>(entity);
}

void RuntimePointComputeTransport::publishProduct(uint64_t socketKey) {
    if (!runtime || socketKey == 0 || !ecsRegistry) {
        return;
    }

    PointProduct product{};
    if (!runtime->exportProduct(socketKey, product)) {
        auto entity = static_cast<ECSEntity>(socketKey);
        ecsRegistry->remove<PointProduct>(entity);
        return;
    }

    auto entity = static_cast<ECSEntity>(socketKey);
    const auto& package = ecsRegistry->get<PointPackage>(entity);
    ecsRegistry->emplace_or_replace<PointProduct>(entity, product);
    appliedConfigInputHash[socketKey] = buildConfigInputHash(socketKey, package);
}
