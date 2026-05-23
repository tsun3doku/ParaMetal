#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"

class ModelDisplayController;

class RuntimeModelDisplayTransport {
public:
    void setController(ModelDisplayController* updatedController) {
        controller = updatedController;
    }

    void setECSRegistry(ECSRegistry* updatedRegistry) {
        ecsRegistry = updatedRegistry;
    }

    void setVisibleKeys(const std::unordered_set<uint64_t>* keys) {
        visibleKeys = keys;
    }

    void sync(const ECSRegistry& registry);
    void finalizeSync();

private:
    ModelDisplayController* controller = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    const std::unordered_set<uint64_t>* visibleKeys = nullptr;
};
