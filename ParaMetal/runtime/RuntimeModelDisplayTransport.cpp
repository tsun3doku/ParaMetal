#include "RuntimeModelDisplayTransport.hpp"

#include "hash/HashBuilder.hpp"
#include "util/GeometryUtils.hpp"
#include "runtime/ModelDisplayController.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "scene/SceneController.hpp"

bool RuntimeModelDisplayTransport::sync(
    const RuntimePackageManager& registry,
    const std::unordered_set<uint64_t>& displayKeys,
    const std::unordered_set<uint64_t>& sceneKeys) {
    if ((!controller && !sceneController) || !products) {
        return true;
    }

    std::unordered_set<uint64_t> nextSocketKeys;
    for (const auto& [socketKey, package] : registry.models()) {
        if (!registry.isPackageCurrent(socketKey)) continue;
        const bool displayed =
            displayKeys.find(socketKey) != displayKeys.end();
        const bool sceneOnly =
            !displayed && sceneKeys.find(socketKey) != sceneKeys.end();
        if (!displayed && !sceneOnly) {
            continue;
        }

        if (!products->isCurrent<ModelProduct>(package.productHandle)) {
            if (controller) controller->remove(socketKey);
            if (sceneController) sceneController->remove(socketKey);
            continue;
        }
        const ModelProduct* product = products->resolve<ModelProduct>(package.productHandle);
        if (!product || product->runtimeModelId == 0) {
            if (controller) controller->remove(socketKey);
            if (sceneController) sceneController->remove(socketKey);
            continue;
        }

        const glm::mat4 modelMatrix = toMat4(package.localToWorld);
        uint64_t displayHash = HashBuilder::start();
        HashBuilder::combine(displayHash, product->runtimeModelId);
        HashBuilder::combinePod(displayHash, modelMatrix);
        HashBuilder::combine(displayHash, product->hashes.display);
        HashBuilder::combine(displayHash, package.productHandle.publicationId);

        if (controller) {
            ModelDisplayController::Config config{};
            config.runtimeModelId = product->runtimeModelId;
            config.modelMatrix = modelMatrix;
            config.displayHash = displayHash;
            const bool applied = displayed
                ? controller->apply(socketKey, config)
                : controller->applySceneOnly(socketKey, config);
            if (!applied) {
                return false;
            }
        }

        if (sceneController) {
            SceneController::Config config{};
            config.runtimeModelId = product->runtimeModelId;
            config.worldBoundsMin = package.worldBoundsMin;
            config.worldBoundsMax = package.worldBoundsMax;
            sceneController->apply(socketKey, config);
        }

        nextSocketKeys.insert(socketKey);
    }

    for (uint64_t socketKey : activeSocketKeys) {
        if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
            if (controller) controller->remove(socketKey);
            if (sceneController) sceneController->remove(socketKey);
        }
    }
    activeSocketKeys = nextSocketKeys;
    return true;
}

bool RuntimeModelDisplayTransport::isDisplayed(uint64_t socketKey) const {
    return controller && controller->isDisplayed(socketKey);
}

bool RuntimeModelDisplayTransport::findByRuntimeModelId(
    uint32_t runtimeModelId,
    uint64_t& outSocketKey) const {
    return controller && controller->findByRuntimeModelId(
        runtimeModelId, outSocketKey);
}

std::vector<uint32_t> RuntimeModelDisplayTransport::registeredRuntimeModelIds() const {
    return controller ? controller->registeredRuntimeModelIds()
                      : std::vector<uint32_t>{};
}

void RuntimeModelDisplayTransport::detach(uint64_t socketKey) {
    if ((!controller && !sceneController) || socketKey == 0) {
        return;
    }
    if (controller) controller->remove(socketKey);
    if (sceneController) sceneController->remove(socketKey);
    activeSocketKeys.erase(socketKey);
}

void RuntimeModelDisplayTransport::detachAll() {
    for (uint64_t socketKey : activeSocketKeys) {
        if (controller) controller->remove(socketKey);
        if (sceneController) sceneController->remove(socketKey);
    }
    activeSocketKeys.clear();
}
