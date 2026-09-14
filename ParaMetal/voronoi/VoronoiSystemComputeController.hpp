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
        bool active = false;
        float cellSize = 0.005f;
        int voxelResolution = 128;
        float contactRange = voronoi::DefaultContactRange;
        std::vector<glm::vec4> pointPositions;
        std::array<glm::vec3, 8> pointDomainCorners{};

        float sdfPadding = 0.0f;
        std::vector<uint32_t> globalRemeshRuntimeModelIds;
        std::vector<std::vector<glm::vec3>> globalRemeshPositions;
        std::vector<std::vector<uint32_t>> globalRemeshTriangleIndices;
        std::vector<std::vector<glm::vec3>> globalRemeshSurfacePositions;
        std::vector<std::vector<uint32_t>> globalRemeshSurfaceTriangleIndices;

        uint64_t computeHash = 0;
    };

    VoronoiSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool);

    void apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, VoronoiProduct& product);
    void remove(uint64_t socketKey);
    void disableAll();
    VoronoiSystem* getSystem(uint64_t socketKey);
    const VoronoiSystem* getSystem(uint64_t socketKey) const;
    bool isConfigured(uint64_t socketKey) const { return configuredHashes.find(socketKey) != configuredHashes.end(); }

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    std::unordered_map<uint64_t, std::unique_ptr<VoronoiSystem>> systemsBySocket;
    std::unordered_map<uint64_t, uint64_t> configuredHashes;
    std::unordered_map<uint64_t, uint64_t> failedComputeHashes;
};
