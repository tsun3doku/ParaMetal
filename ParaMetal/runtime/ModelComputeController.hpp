#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>

#include "hash/HashValues.hpp"
#include "runtime/RuntimeProducts.hpp"

class ModelUploader;
class ModelRegistry;

class ModelComputeController {
public:
    struct Config {
        std::vector<float> vertexPositions;
        std::vector<uint32_t> triangleIndices;
        std::vector<float> renderPositions;
        std::vector<float> renderNormals;
        std::vector<float> renderTexcoords;
        std::vector<uint32_t> renderIndices;
        uint64_t computeHash = 0;
    };

    ModelComputeController(
        ModelRegistry& modelRegistry,
        ModelUploader& modelUploader,
        std::atomic<bool>& isOperating);

    bool apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, ModelProduct& product) const;
    bool commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    bool remove(uint64_t socketKey);
    void disableAll();
    bool hasAnyRetiring() const;
    void clearAllRetiring();
    uint64_t getCurrentComputeHash(uint64_t socketKey) const;

private:
    class OperatingScope {
    public:
        explicit OperatingScope(std::atomic<bool>& isOperating);
        ~OperatingScope();

    private:
        std::atomic<bool>& isOperating;
        bool previousState = false;
    };

    struct PendingModel {
        uint32_t runtimeModelId = 0;
        uint64_t computeHash = 0;
    };

    bool prepareModel(uint64_t socketKey, const Config& config);
    bool removeModel(uint64_t socketKey);

    ModelRegistry& modelRegistry;
    ModelUploader& modelUploader;
    std::atomic<bool>& isOperating;
    mutable std::mutex executionMutex;
    std::unordered_map<uint64_t, uint32_t> runtimeModelIdBySocketKey;
    std::unordered_map<uint64_t, uint64_t> configuredComputeHashes;
    std::unordered_map<uint64_t, PendingModel> pendingModels;
};
