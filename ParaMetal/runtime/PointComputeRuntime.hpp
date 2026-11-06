#pragma once

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "runtime/RuntimeProducts.hpp"
#include "util/Units.hpp"

class MemoryAllocator;

class PointComputeRuntime {
public:
    struct Config {
        uint64_t socketKey = 0;
        std::vector<glm::vec4> positions;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        uint64_t computeHash = 0;
    };

    explicit PointComputeRuntime(MemoryAllocator& memoryAllocator);
    ~PointComputeRuntime();

    // Allocates the candidate generation. The active generation stays live.
    bool prepareCandidate(const Config& config);
    // Read-only view of the candidate when present, otherwise the current generation.
    bool buildProduct(uint64_t socketKey, PointProduct& product) const;
    void commitPending(uint64_t socketKey) noexcept;
    void cancelCandidate(uint64_t socketKey);

    void clearRetiring(uint64_t socketKey);

    void disable(uint64_t socketKey);
    bool hasAnyRetiring() const;
    void clearAllRetiring();
    // True when an active generation already satisfies the requested compute hash.
    bool currentHashMatches(uint64_t socketKey, uint64_t computeHash) const;
    // Unconditional release used during quiesced teardown.
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct PointBufferSlot {
        std::pair<VkBuffer, VkDeviceSize> current{VK_NULL_HANDLE, 0};
        std::pair<VkBuffer, VkDeviceSize> retiring{VK_NULL_HANDLE, 0};
        std::pair<VkBuffer, VkDeviceSize> candidate{VK_NULL_HANDLE, 0};
        uint64_t currentHash = 0;
        uint64_t candidateHash = 0;
        uint32_t currentPointCount = 0;
        uint32_t candidatePointCount = 0;
        glm::mat4 currentModelMatrix = glm::mat4(1.0f);
        glm::mat4 candidateModelMatrix = glm::mat4(1.0f);
        bool candidateValid = false;
    };

    void releaseBuffer(std::pair<VkBuffer, VkDeviceSize>& buffer);
    void releaseSlot(PointBufferSlot& slot);

    MemoryAllocator& memoryAllocator;
    std::unordered_map<uint64_t, PointBufferSlot> slots;
};
