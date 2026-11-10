#pragma once

#include "cfd/CFDSystem.hpp"
#include <unordered_map>
#include <memory>
#include <vector>
#include <cstdint>

struct CFDProduct;

namespace cfd {

class CFDSystemComputeController {
public:
    struct Config {
        const topology::GlobalDomain* domain = nullptr;
        float kinematicViscosity = 1.516e-5f;
        glm::vec3 initialVelocity{0.0f};
        uint32_t domainBoundaryTypes[6] = {};
        glm::vec3 domainBoundaryVelocities[6] = {};
        float fixedTimeStep = 1.0f / 60.0f;
        float targetCourant = 0.5f;
        int pressureIterations = 0;
        float pressureTolerance = 0.0f;
        float simulationDuration = 10.0f;
        uint64_t computeHash = 0;
        uint64_t geometryHash = 0;
    };

    static bool canUpdateInPlace(const Config& previous, const Config& next);

    CFDSystemComputeController(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        CommandPool& cmdPool,
        uint32_t maxFramesInFlight);
    ~CFDSystemComputeController();

    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);

    void releaseWaits();

    bool buildProduct(uint64_t socketKey, CFDProduct& outProduct);
    const CFDSystem* getSystem(uint64_t socketKey) const;
    CFDSystem* getSystem(uint64_t socketKey);
    const Config* getConfig(uint64_t socketKey) const;
    uint64_t getCurrentComputeHash(uint64_t socketKey) const {
        const auto it = configuredConfigs.find(socketKey);
        return it != configuredConfigs.end() ? it->second.computeHash : 0;
    }

    bool hasAnyRetiring() const;
    void clearAllRetiring();
    void commitPending(uint64_t socketKey);
    void cancelCandidate(uint64_t socketKey);
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct GenerationSlot {
        std::unique_ptr<CFDSystem> current;
        std::unique_ptr<CFDSystem> retiring;
        std::unique_ptr<CFDSystem> candidate;
    };

    const Config* getBuildConfig(uint64_t socketKey) const;

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight;

    std::unordered_map<uint64_t, GenerationSlot> generationSlots;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    std::unordered_map<uint64_t, Config> pendingConfigs;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
};

} // namespace cfd
