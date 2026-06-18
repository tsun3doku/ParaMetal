#pragma once

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <unordered_map>
#include <unordered_set>

#include "hash/HashBuilder.hpp"

class ModelRegistry;

class ModelDisplayController {
public:
    struct Config {
        uint32_t runtimeModelId = 0;
        glm::mat4 modelMatrix{ 1.0f };
        uint64_t displayHash = 0;
    };

    void setModelRegistry(ModelRegistry* registry);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    ModelRegistry* modelRegistry = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};
