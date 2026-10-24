#pragma once

#include "cfd/CFDSystem.hpp"
#include "runtime/TimelinePlaybackTarget.hpp"
#include <unordered_map>
#include <memory>
#include <vector>
#include <cstdint>

struct CFDProduct;

namespace cfd {

class CFDSystemComputeController : public TimelinePlaybackTarget {
public:
    struct Config {
        bool active = true;
        const topology::GlobalDomain* domain = nullptr;
        float kinematicViscosity = 1.516e-5f;
        glm::vec3 initialVelocity{0.0f};
        uint32_t domainBoundaryTypes[6] = {};
        glm::vec3 domainBoundaryVelocities[6] = {};
        float fixedTimeStep = 1.0f / 60.0f;
        float targetCourant = 0.5f;
        int pressureIterations = 0;
        float pressureTolerance = 0.0f;
        uint64_t computeHash = 0;
        uint64_t geometryHash = 0;
    };

    // Tier 2 eligibility check shared by apply and transport domain-skip logic
    static bool canUpdateInPlace(const Config& previous, const Config& next);

    CFDSystemComputeController(
        VulkanDevice& device,
        CommandPool& cmdPool,
        uint32_t maxFramesInFlight);
    ~CFDSystemComputeController() override;

    // Returns false on failure, previous generation stays active and caller skips publish
    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void disableAll();

    // Host-signals pending Vulkan release waits for current and retained systems
    void releaseWaits();

    void setTimelinePlaying(bool playing) override;
    void resetTimeline() override;
    void scrubTimeline(uint32_t frame) override;

    bool buildProduct(uint64_t socketKey, CFDProduct& outProduct);
    const CFDSystem* getSystem(uint64_t socketKey) const;
    CFDSystem* getSystem(uint64_t socketKey);
    const Config* getConfig(uint64_t socketKey) const;

private:
    void applyRuntimeState(CFDSystem& system, const Config& config);

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight;

    bool timelinePlaying = false;

    std::unordered_map<uint64_t, std::unique_ptr<CFDSystem>> systemsBySocket;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
    std::vector<std::unique_ptr<CFDSystem>> retainedSystems;
};

} // namespace cfd
