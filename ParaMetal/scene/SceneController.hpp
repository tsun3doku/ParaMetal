#pragma once

#include <cstdint>
#include <unordered_map>

#include <glm/vec3.hpp>

class SceneController {
public:
    struct Config {
        uint32_t runtimeModelId = 0;
        glm::vec3 worldBoundsMin{0.0f};
        glm::vec3 worldBoundsMax{0.0f};
    };

    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);

    bool tryGetRuntimeModelSocketKey(
        uint32_t runtimeModelId,
        uint64_t& outSocketKey) const;

    glm::vec3 sceneWorldExtent() const;

private:
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_map<uint32_t, uint64_t> socketKeyByRuntimeModelId;
};
