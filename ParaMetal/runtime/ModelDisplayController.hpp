#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <unordered_map>
#include <vector>

#include "hash/HashBuilder.hpp"

class ModelRegistry;

class ModelDisplayController {
public:
    struct Config {
        uint32_t runtimeModelId = 0;
        glm::mat4 modelMatrix{ 1.0f };
        uint64_t displayHash = 0;
        bool displayed = false;
    };

    void setModelRegistry(ModelRegistry* registry);
    bool apply(uint64_t socketKey, const Config& config);
    bool applySceneOnly(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    bool isDisplayed(uint64_t socketKey) const;
    bool findByRuntimeModelId(
        uint32_t runtimeModelId,
        uint64_t& outSocketKey) const;
    std::vector<uint32_t> registeredRuntimeModelIds() const;

private:
    ModelRegistry* modelRegistry = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
};
