#pragma once

#include "voronoi/VoronoiSystem.hpp"
#include "voronoi/VoronoiConfig.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/Units.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

class MemoryAllocator;
class VulkanDevice;
class CommandPool;
struct VoronoiProduct;

class VoronoiSystemComputeController {
public:
    struct Config {
        float cellSize = 0.005f;
        int voxelResolution = 128;
        int lloydIterations = int(voronoi::DefaultLloydIterations);
        float lloydVolumeThreshold = voronoi::DefaultLloydVolumeThresholdFraction;
        float lloydDamping = voronoi::DefaultLloydDamping;
        float contactRange = voronoi::DefaultContactRange;
        std::vector<glm::vec4> pointPositions;
        std::array<glm::vec3, 8> pointDomainCorners{};

        float volumePadding = 0.0f;
        std::vector<uint32_t> globalRemeshRuntimeModelIds;
        std::vector<std::vector<glm::vec3>> globalRemeshPositions;
        std::vector<std::vector<uint32_t>> globalRemeshTriangleIndices;
        std::vector<std::vector<glm::vec3>> globalRemeshSurfacePositions;
        std::vector<std::vector<uint32_t>> globalRemeshSurfaceTriangleIndices;
        std::vector<std::array<float, 16>> globalRemeshWorldToModel;

        uint64_t computeHash = 0;
    };

    VoronoiSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool);

    bool apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, VoronoiProduct& product);
    void remove(uint64_t socketKey);
    VoronoiSystem* getSystem(uint64_t socketKey);

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
        std::unique_ptr<VoronoiSystem> current;
        std::unique_ptr<VoronoiSystem> retiring;
        std::unique_ptr<VoronoiSystem> candidate;
    };

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    std::unordered_map<uint64_t, GenerationSlot> generationSlots;
    std::unordered_map<uint64_t, uint64_t> configuredHashes;
    std::unordered_map<uint64_t, uint64_t> pendingHashes;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
};
