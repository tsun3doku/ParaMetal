#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "mesh/remesher/Remesher.hpp"
#include "runtime/RemeshSystem.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/Units.hpp"

class MemoryAllocator;
class VulkanDevice;

struct RemeshProduct;

class RemeshController {
public:
    struct Config {
        std::vector<float> pointPositions;
        std::vector<uint32_t> triangleIndices;
        int iterations = 1;
        float minAngleDegrees = 20.0f;
        float maxEdgeLength = 0.1f;
        float stepSize = 0.25f;
        uint32_t runtimeModelId = 0;
        uint64_t computeHash = 0;
    };

    RemeshController(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, std::atomic<bool>& isOperating);

    // Returns false on failure, previous generation stays active and caller skips publish
    bool apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, RemeshProduct& product);
    void remove(uint64_t socketKey);
    uint64_t getCurrentComputeHash(uint64_t socketKey) const {
        const auto it = configuredHashes.find(socketKey);
        return it != configuredHashes.end() ? it->second : 0;
    }

    bool hasAnyRetiring() const;
    void clearAllRetiring();
    void commitPending(uint64_t socketKey) noexcept;
    void cancelCandidate(uint64_t socketKey);
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct GenerationSlot {
        std::unique_ptr<RemeshSystem> current;
        std::unique_ptr<RemeshSystem> retiring;
        std::unique_ptr<RemeshSystem> candidate;
    };

    class OperatingScope {
    public:
        explicit OperatingScope(std::atomic<bool>& isOperating);
        ~OperatingScope();

    private:
        std::atomic<bool>& isOperating;
        bool previousState = false;
    };

    std::atomic<bool>& isOperating;
    Remesher remesher;
    std::unordered_map<uint64_t, GenerationSlot> generationSlots;
    std::unordered_map<uint64_t, uint64_t> configuredHashes;
    std::unordered_map<uint64_t, uint64_t> pendingHashes;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
};
