#pragma once

#include "HeatSystemPresets.hpp"
#include "heat/VoronoiSystem.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"

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
        uint32_t receiverNodeModelId = 0;
        std::vector<glm::vec3> receiverGeometryPositions;
        std::vector<uint32_t> receiverGeometryTriangleIndices;
        SupportingHalfedge::IntrinsicMesh receiverIntrinsicMesh;
        std::vector<VoronoiModelRuntime::SurfaceVertex> receiverSurfaceVertices;
        std::vector<uint32_t> receiverIntrinsicTriangleIndices;
        uint32_t receiverRuntimeModelId = 0;
        glm::mat4 meshModelMatrix{1.0f};
        std::vector<VkBuffer> receiverSurfaceBuffers;
        std::vector<VkDeviceSize> receiverSurfaceBufferOffsets;
        std::vector<VkBuffer> receiverSurfaceGradientBuffers;
        std::vector<VkDeviceSize> receiverSurfaceGradientBufferOffsets;
        uint64_t computeHash = 0;
    };

    VoronoiSystemComputeController(
        VulkanDevice& vulkanDevice, 
        MemoryAllocator& memoryAllocator,
        ModelRegistry& resourceManager,
        CommandPool& renderCommandPool,
        uint32_t maxFramesInFlight);

    void configure(uint64_t socketKey, const Config& config);
    void disable(uint64_t socketKey);
    void disableAll();
    const VoronoiSystem* getSystem(uint64_t socketKey) const;
    const Config* getConfig(uint64_t socketKey) const;
    std::vector<VoronoiSystem*> getActiveSystems() const;

private:
    std::unique_ptr<VoronoiSystem> buildVoronoiSystem();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& renderCommandPool;
    std::unordered_map<uint64_t, std::unique_ptr<VoronoiSystem>> activeSystems;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    uint32_t maxFramesInFlight = 0;
};

inline uint64_t buildComputeHash(const VoronoiSystemComputeController::Config& config) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.active ? 1u : 0u));
    NodeGraphHash::combinePod(hash, config.cellSize);
    NodeGraphHash::combinePod(hash, config.voxelResolution);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.isPointDomain ? 1u : 0u));
    NodeGraphHash::combinePodVector(hash, config.pointPositions);
    NodeGraphHash::combine(hash, config.receiverNodeModelId);
    NodeGraphHash::combinePodVector(hash, config.receiverGeometryPositions);
    NodeGraphHash::combinePodVector(hash, config.receiverGeometryTriangleIndices);
    NodeGraphHash::combinePodVector(hash, config.receiverIntrinsicMesh.vertices);
    NodeGraphHash::combinePodVector(hash, config.receiverIntrinsicMesh.indices);
    NodeGraphHash::combinePodVector(hash, config.receiverIntrinsicMesh.faceIds);
    NodeGraphHash::combinePodVector(hash, config.receiverIntrinsicMesh.triangles);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceVertices);
    NodeGraphHash::combinePodVector(hash, config.receiverIntrinsicTriangleIndices);
    NodeGraphHash::combine(hash, config.receiverRuntimeModelId);
    NodeGraphHash::combinePod(hash, config.meshModelMatrix);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceBuffers);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceGradientBuffers);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceGradientBufferOffsets);
    return hash;
}
