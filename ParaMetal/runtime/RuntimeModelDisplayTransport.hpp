#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

class ModelDisplayController;
class SceneController;

class RuntimeModelDisplayTransport {
public:
    void setController(ModelDisplayController* updatedController) {
        controller = updatedController;
    }

    void setSceneController(SceneController* updatedController) {
        sceneController = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    bool sync(
        const RuntimePackageManager& registry,
        const std::unordered_set<uint64_t>& displayKeys,
        const std::unordered_set<uint64_t>& sceneKeys);
    bool isDisplayed(uint64_t socketKey) const;
    bool findByRuntimeModelId(
        uint32_t runtimeModelId,
        uint64_t& outSocketKey) const;
    std::vector<uint32_t> registeredRuntimeModelIds() const;

    // Deterministic detach used by the removal protocol before a backend retires.
    void detach(uint64_t socketKey);
    void detachAll();

private:
    ModelDisplayController* controller = nullptr;
    SceneController* sceneController = nullptr;
    RuntimeProductManager* products = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
