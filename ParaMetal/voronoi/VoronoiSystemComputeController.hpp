#pragma once

#include "voronoi/VoronoiSystem.hpp"
#include "runtime/RuntimeProducts.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>
#include <vulkan/vulkan.h>

class MemoryAllocator;
class ModelRegistry;
class VulkanDevice;
class CommandPool;
struct VoronoiProduct;

class VoronoiSystemComputeController {
public:
    struct Config {
        bool active = false;
        float cellSize = 0.005f;
        int voxelResolution = 128;
        bool isPointDomain = false;
        std::vector<glm::vec4> pointPositions;

        // Mesh path
        std::vector<glm::vec3> geometryPositions;
        std::vector<uint32_t> geometryTriangleIndices;
        std::vector<voronoi::SurfaceVertex> surfaceVertices;
        std::vector<uint32_t> surfaceTriangleIndices;
        uint32_t runtimeModelId = 0;
        glm::mat4 meshModelMatrix{1.0f};
        uint64_t computeHash = 0;
    };

    VoronoiSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        ModelRegistry& resourceManager,
        CommandPool& commandPool,
        uint32_t maxFramesInFlight);

    void apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, VoronoiProduct& product) const;
    void remove(uint64_t socketKey);
    void disableAll();
    std::vector<VoronoiSystem*> getActiveSystems() const;
    const VoronoiSystem* getSystem(uint64_t socketKey) const;
    const Config* getConfig(uint64_t socketKey) const;

private:
    std::unique_ptr<VoronoiSystem> buildVoronoiSystem();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& commandPool;
    std::unordered_map<uint64_t, std::unique_ptr<VoronoiSystem>> systemsBySocket;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    uint32_t maxFramesInFlight = 0;
};
