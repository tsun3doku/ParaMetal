#include "RuntimeModelDisplayTransport.hpp"

#include "hash/HashBuilder.hpp"
#include "util/GeometryUtils.hpp"
#include "runtime/ModelDisplayController.hpp"
#include "runtime/RuntimePackageManager.hpp"

void RuntimeModelDisplayTransport::sync(const RuntimePackageManager& registry, const std::unordered_set<uint64_t>& visibleKeys) {
    if (!controller || !products) {
        return;
    }

    std::unordered_set<uint64_t> nextSocketKeys;
    registry.forEach<ModelPackage>([&](uint64_t socketKey, const ModelPackage& package) {
        if (visibleKeys.find(socketKey) == visibleKeys.end()) {
            return;
        }

        const ModelProduct* product = products->resolve<ModelProduct>(package.productHandle);
        if (!product || product->runtimeModelId == 0) {
            controller->remove(socketKey);
            return;
        }

        ModelDisplayController::Config config{};
        config.runtimeModelId = product->runtimeModelId;
        config.modelMatrix = toMat4(package.localToWorld);
        uint64_t displayHash = HashBuilder::start();
        HashBuilder::combine(displayHash, config.runtimeModelId);
        HashBuilder::combinePod(displayHash, config.modelMatrix);
        HashBuilder::combine(displayHash, product->hashes.display);
        config.displayHash = displayHash;
        controller->apply(socketKey, config);
        nextSocketKeys.insert(socketKey);
    });

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
