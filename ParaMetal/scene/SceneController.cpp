#include "SceneController.hpp"

#include <glm/common.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

void SceneController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0 || config.runtimeModelId == 0) {
        return;
    }

    const auto reverseIt =
        socketKeyByRuntimeModelId.find(config.runtimeModelId);
    if (reverseIt != socketKeyByRuntimeModelId.end() &&
        reverseIt->second != socketKey) {
        std::terminate();
    }

    const auto existingIt = configsBySocket.find(socketKey);
    if (existingIt != configsBySocket.end()) {
        const auto oldRuntimeIt =
            socketKeyByRuntimeModelId.find(existingIt->second.runtimeModelId);
        if (oldRuntimeIt != socketKeyByRuntimeModelId.end() &&
            oldRuntimeIt->second == socketKey) {
            socketKeyByRuntimeModelId.erase(oldRuntimeIt);
        }
    }

    configsBySocket[socketKey] = config;
    socketKeyByRuntimeModelId[config.runtimeModelId] = socketKey;
}

void SceneController::remove(uint64_t socketKey) {
    const auto it = configsBySocket.find(socketKey);
    if (it == configsBySocket.end()) {
        return;
    }

    const auto runtimeIt =
        socketKeyByRuntimeModelId.find(it->second.runtimeModelId);
    if (runtimeIt != socketKeyByRuntimeModelId.end() &&
        runtimeIt->second == socketKey) {
        socketKeyByRuntimeModelId.erase(runtimeIt);
    }
    configsBySocket.erase(it);
}

bool SceneController::tryGetRuntimeModelSocketKey(
    uint32_t runtimeModelId,
    uint64_t& outSocketKey) const {
    outSocketKey = 0;
    const auto it = socketKeyByRuntimeModelId.find(runtimeModelId);
    if (it == socketKeyByRuntimeModelId.end()) {
        return false;
    }
    outSocketKey = it->second;
    return outSocketKey != 0;
}

glm::vec3 SceneController::sceneWorldExtent() const {
    glm::vec3 worldMin(std::numeric_limits<float>::max());
    glm::vec3 worldMax(std::numeric_limits<float>::lowest());
    bool hasBounds = false;
    for (const auto& entry : configsBySocket) {
        worldMin = glm::min(worldMin, entry.second.worldBoundsMin);
        worldMax = glm::max(worldMax, entry.second.worldBoundsMax);
        hasBounds = true;
    }
    if (!hasBounds) {
        return glm::vec3(1.0f);
    }

    const glm::vec3 boundsSize = worldMax - worldMin;
    const float largestExtent =
        std::max(boundsSize.x, std::max(boundsSize.y, boundsSize.z));
    if (!(largestExtent > 0.0f) || !std::isfinite(largestExtent)) {
        return glm::vec3(1.0f);
    }

    const float padding = largestExtent * 0.25f;
    const glm::vec3 gridAxesSize(boundsSize.x, boundsSize.z, boundsSize.y);
    return glm::max(
        gridAxesSize + glm::vec3(padding),
        glm::vec3(largestExtent * 0.5f));
}

