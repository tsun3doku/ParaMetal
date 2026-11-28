#include "ModelDisplayController.hpp"

#include "vulkan/ModelRegistry.hpp"

#include <algorithm>

void ModelDisplayController::setModelRegistry(ModelRegistry* registry) {
    modelRegistry = registry;
}

bool ModelDisplayController::apply(uint64_t socketKey, const Config& config) {
    if (!modelRegistry || socketKey == 0 || config.runtimeModelId == 0) {
        return false;
    }

    const auto existingIt = configsBySocket.find(socketKey);
    if (existingIt != configsBySocket.end() &&
        existingIt->second.displayHash == config.displayHash &&
        existingIt->second.displayed) {
        return true;
    }
    if (!modelRegistry->hasModel(config.runtimeModelId)) {
        return false;
    }

    const uint32_t previousModelId =
        existingIt != configsBySocket.end()
            ? existingIt->second.runtimeModelId
            : 0;
    if (previousModelId != 0 &&
        previousModelId != config.runtimeModelId &&
        !modelRegistry->hasModel(previousModelId)) {
        return false;
    }

    if (!modelRegistry->setModelMatrix(
            config.runtimeModelId, config.modelMatrix)) {
        return false;
    }
    if (!modelRegistry->setModelVisible(config.runtimeModelId, true)) {
        return false;
    }
    if (previousModelId != 0 &&
        previousModelId != config.runtimeModelId &&
        !modelRegistry->setModelVisible(previousModelId, false)) {
        modelRegistry->setModelVisible(config.runtimeModelId, false);
        return false;
    }

    Config displayedConfig = config;
    displayedConfig.displayed = true;
    configsBySocket[socketKey] = displayedConfig;
    return true;
}

bool ModelDisplayController::applySceneOnly(
    uint64_t socketKey,
    const Config& config) {
    if (!modelRegistry || socketKey == 0 || config.runtimeModelId == 0) {
        return false;
    }

    const auto existingIt = configsBySocket.find(socketKey);
    if (existingIt != configsBySocket.end() &&
        existingIt->second.displayHash == config.displayHash &&
        !existingIt->second.displayed) {
        return true;
    }

    if (!modelRegistry->hasModel(config.runtimeModelId)) {
        return false;
    }
    if (!modelRegistry->setModelMatrix(
            config.runtimeModelId, config.modelMatrix)) {
        return false;
    }
    if (!modelRegistry->setModelVisible(config.runtimeModelId, false)) {
        return false;
    }

    if (existingIt != configsBySocket.end() &&
        existingIt->second.runtimeModelId != config.runtimeModelId &&
        modelRegistry->hasModel(existingIt->second.runtimeModelId)) {
        modelRegistry->setModelVisible(existingIt->second.runtimeModelId, false);
    }

    Config sceneConfig = config;
    sceneConfig.displayed = false;
    configsBySocket[socketKey] = sceneConfig;
    return true;
}

void ModelDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    const auto it = configsBySocket.find(socketKey);
    if (it != configsBySocket.end()) {
        if (it->second.displayed && modelRegistry) {
            modelRegistry->setModelVisible(it->second.runtimeModelId, false);
        }
        configsBySocket.erase(it);
    }
}

bool ModelDisplayController::isDisplayed(uint64_t socketKey) const {
    const auto it = configsBySocket.find(socketKey);
    return it != configsBySocket.end() && it->second.displayed;
}

bool ModelDisplayController::findByRuntimeModelId(
    uint32_t runtimeModelId,
    uint64_t& outSocketKey) const {
    outSocketKey = 0;
    if (runtimeModelId == 0) {
        return false;
    }
    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.displayed || config.runtimeModelId != runtimeModelId) {
            continue;
        }
        if (outSocketKey != 0) {
            outSocketKey = 0;
            return false;
        }
        outSocketKey = socketKey;
    }
    return outSocketKey != 0;
}

std::vector<uint32_t> ModelDisplayController::registeredRuntimeModelIds() const {
    std::vector<uint32_t> ids;
    ids.reserve(configsBySocket.size());
    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (!config.displayed || config.runtimeModelId == 0) {
            continue;
        }
        if (std::find(ids.begin(), ids.end(), config.runtimeModelId) == ids.end()) {
            ids.push_back(config.runtimeModelId);
        }
    }
    return ids;
}
