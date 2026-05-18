#include "RuntimeModelDisplayTransport.hpp"

#include "nodegraph/NodeModelTransform.hpp"
#include "runtime/ModelDisplayController.hpp"
#include "runtime/RuntimeECS.hpp"

void RuntimeModelDisplayTransport::sync(const ECSRegistry& registry) {
    if (!controller) {
        return;
    }

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
        config.modelMatrix = NodeModelTransform::toMat4(package.localToWorld);
        controller->apply(socketKey, config);
    }

    for (auto entity : registry.view<ModelPackage, Stale>()) {
        uint64_t socketKey = static_cast<uint64_t>(entity);
        controller->remove(socketKey);
    }
}

void RuntimeModelDisplayTransport::finalizeSync() {
    if (!controller) {
        return;
    }

    controller->finalizeSync();
}
