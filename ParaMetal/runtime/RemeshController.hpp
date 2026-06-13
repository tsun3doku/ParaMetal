#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mesh/remesher/Remesher.hpp"
#include "runtime/RemeshSystem.hpp"
#include "runtime/RuntimeProducts.hpp"

class MemoryAllocator;
class ModelRegistry;
class VulkanDevice;

class RemeshController {
public:
    struct Config {
        uint64_t socketKey = 0;
        std::vector<float> pointPositions;
        std::vector<uint32_t> triangleIndices;
        int iterations = 1;
        float minAngleDegrees = 20.0f;
        float maxEdgeLength = 0.1f;
        float stepSize = 0.25f;
        uint32_t runtimeModelId = 0;
        uint64_t computeHash = 0;
    };

    RemeshController(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, ModelRegistry& resourceManager, std::atomic<bool>& isOperating);

    void configure(const Config& config);
    void disable(uint64_t socketKey);
    void disable();
    const RemeshSystem* getSystem(uint64_t socketKey) const;

    Remesher& getRemesher() { return remesher; }
    const Remesher& getRemesher() const { return remesher; }

private:
    class OperatingScope {
    public:
        explicit OperatingScope(std::atomic<bool>& isOperating);
        ~OperatingScope();

    private:
        std::atomic<bool>& isOperating;
        bool previousState = false;
    };

    VulkanDevice& vulkanDevice;
    ModelRegistry& resourceManager;
    std::atomic<bool>& isOperating;
    Remesher remesher;
    std::unordered_map<uint64_t, std::unique_ptr<RemeshSystem>> activeSystems;
    std::unordered_map<uint64_t, Config> configuredConfigs;
};
