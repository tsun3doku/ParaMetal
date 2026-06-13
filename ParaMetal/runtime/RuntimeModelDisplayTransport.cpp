#include "RuntimeModelDisplayTransport.hpp"

#include "nodegraph/NodeGraphHash.hpp"
#include "util/GeometryUtils.hpp"
#include "runtime/ModelDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"

void RuntimeModelDisplayTransport::sync(const ECSRegistry& registry) {
    if (!controller) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;
    auto view = registry.view<ModelPackage>(entt::exclude<Stale>);
    for (auto entity : view) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        if (visibleKeys && visibleKeys->find(socketKey) == visibleKeys->end()) {
            continue;
        }

        const auto& package = registry.get<ModelPackage>(entity);
        const ModelProduct* product = tryGetProduct<ModelProduct>(registry, socketKey);

        ModelDisplayController::Config config{};
        config.runtimeModelId = product ? product->runtimeModelId : 0;
        config.modelMatrix = toMat4(package.localToWorld);
        uint64_t displayHash = NodeGraphHash::start();
        NodeGraphHash::combine(displayHash, config.runtimeModelId);
        NodeGraphHash::combinePod(displayHash, config.modelMatrix);
        if (product) {
            NodeGraphHash::combine(displayHash, product->productHash);
        }
        config.displayHash = displayHash;
        controller->apply(socketKey, config);
        nextSocketKeys.insert(socketKey);
    }

    for (auto entity : registry.view<ModelPackage, Stale>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        controller->remove(socketKey);
    }

    for (uint64_t socketKey : activeSocketKeys) {
        if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
            controller->remove(socketKey);
        }
    }
    activeSocketKeys = std::move(nextSocketKeys);
}

void RuntimeModelDisplayTransport::finalizeSync() {
    if (!controller) {
        return;
    }

    controller->finalizeSync();
}
