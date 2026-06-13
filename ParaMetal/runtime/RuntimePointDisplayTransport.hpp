#pragma once

#include "runtime/PointDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"

#include <unordered_set>

class RuntimePointDisplayTransport {
public:
    void setController(PointDisplayController* updatedController) {
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
        auto view = registry.view<PointPackage>(entt::exclude<Stale>);
        for (auto entity : view) {
            uint64_t socketKey = static_cast<uint64_t>(entity);
            if (visibleKeys && visibleKeys->find(socketKey) == visibleKeys->end()) {
                continue;
            }

            const auto& package = registry.get<PointPackage>(entity);
            PointDisplayController::Config config{};
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
    bool tryBuildConfig(uint64_t socketKey, const PointPackage& package, PointDisplayController::Config& outConfig) const {
        if (!controller || !ecsRegistry || socketKey == 0) {
            return false;
        }

        const PointProduct* product = tryGetProduct<PointProduct>(*ecsRegistry, socketKey);
        if (!product || !product->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.vertexBuffer = product->positionBuffer;
        outConfig.vertexBufferOffset = product->positionBufferOffset;
        outConfig.pointCount = product->pointCount;
        outConfig.modelMatrix = product->modelMatrix;
        outConfig.displayHash = product->productHash;
        return true;
    }

    PointDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
