#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "runtime/RuntimeProducts.hpp"
#include "util/Units.hpp"

class MemoryAllocator;

class PointComputeRuntime {
public:
    struct Config {
        uint64_t socketKey = 0;
        const glm::vec4* positions = nullptr;
        std::size_t positionCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        uint64_t computeHash = 0;
    };

    explicit PointComputeRuntime(MemoryAllocator& memoryAllocator);
    ~PointComputeRuntime();

    bool prepareCandidate(const Config& config);
    bool buildProduct(uint64_t socketKey, PointProduct& product) const;
    void commitPending(uint64_t socketKey) noexcept;
    void cancelCandidate(uint64_t socketKey);

    void clearRetiring(uint64_t socketKey);

    void disable(uint64_t socketKey);
    bool hasAnyRetiring() const;
    void clearAllRetiring();
    bool currentHashMatches(uint64_t socketKey, uint64_t computeHash) const;
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct PointBufferGeneration {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        uint64_t hash = 0;
        uint32_t pointCount = 0;
        glm::mat4 modelMatrix{1.0f};
    };

    struct PointBufferSlot {
        PointBufferGeneration current;
        PointBufferGeneration retiring;
        PointBufferGeneration candidate;
    };

    void releaseGeneration(PointBufferGeneration& generation);
    void releaseSlot(PointBufferSlot& slot);

    MemoryAllocator& memoryAllocator;
    std::unordered_map<uint64_t, PointBufferSlot> slots;
};
