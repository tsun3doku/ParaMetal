#include "ModelDisplayController.hpp"

#include "vulkan/ModelRegistry.hpp"

#include <vector>

void ModelDisplayController::setModelRegistry(ModelRegistry* registry) {
    modelRegistry = registry;
}

void ModelDisplayController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0 || config.runtimeModelId == 0) {
        return;
    }

    syncedSockets.insert(socketKey);

    const auto existingIt = configsBySocket.find(socketKey);
    if (existingIt != configsBySocket.end() && existingIt->second.displayHash == config.displayHash) {
        return;
    }

    if (modelRegistry) {
        modelRegistry->setModelVisible(config.runtimeModelId, true);
        modelRegistry->setModelMatrix(config.runtimeModelId, config.modelMatrix);
    }

    configsBySocket[socketKey] = config;
}

void ModelDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    const auto it = configsBySocket.find(socketKey);
    if (it != configsBySocket.end() && modelRegistry) {
        modelRegistry->setModelVisible(it->second.runtimeModelId, false);
    }

    configsBySocket.erase(socketKey);
}

void ModelDisplayController::finalizeSync() {
    std::vector<uint64_t> staleSocketKeys;
    staleSocketKeys.reserve(configsBySocket.size());
    for (const auto& [socketKey, config] : configsBySocket) {
        (void)config;
        if (syncedSockets.find(socketKey) == syncedSockets.end()) {
            staleSocketKeys.push_back(socketKey);
        }
    }

    for (uint64_t socketKey : staleSocketKeys) {
        remove(socketKey);
    }

    syncedSockets.clear();
}
