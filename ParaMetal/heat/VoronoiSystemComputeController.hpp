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
        std::vector<uint32_t> receiverNodeModelIds;
        std::vector<std::vector<glm::vec3>> receiverGeometryPositions;
        std::vector<std::vector<uint32_t>> receiverGeometryTriangleIndices;
        std::vector<SupportingHalfedge::IntrinsicMesh> receiverIntrinsicMeshes;
        std::vector<std::vector<VoronoiModelRuntime::SurfaceVertex>> receiverSurfaceVertices;
        std::vector<std::vector<uint32_t>> receiverIntrinsicTriangleIndices;
        std::vector<uint32_t> receiverRuntimeModelIds;
        std::vector<glm::mat4> meshModelMatrices;
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
    NodeGraphHash::combinePodVector(hash, config.receiverNodeModelIds);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.receiverGeometryPositions.size()));
    for (const auto& positions : config.receiverGeometryPositions) {
        NodeGraphHash::combinePodVector(hash, positions);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.receiverGeometryTriangleIndices.size()));
    for (const auto& indices : config.receiverGeometryTriangleIndices) {
        NodeGraphHash::combinePodVector(hash, indices);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.receiverIntrinsicMeshes.size()));
    for (const SupportingHalfedge::IntrinsicMesh& mesh : config.receiverIntrinsicMeshes) {
        NodeGraphHash::combinePodVector(hash, mesh.vertices);
        NodeGraphHash::combinePodVector(hash, mesh.indices);
        NodeGraphHash::combinePodVector(hash, mesh.faceIds);
        NodeGraphHash::combinePodVector(hash, mesh.triangles);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.receiverSurfaceVertices.size()));
    for (const auto& vertices : config.receiverSurfaceVertices) {
        NodeGraphHash::combinePodVector(hash, vertices);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(config.receiverIntrinsicTriangleIndices.size()));
    for (const auto& indices : config.receiverIntrinsicTriangleIndices) {
        NodeGraphHash::combinePodVector(hash, indices);
    }
    NodeGraphHash::combinePodVector(hash, config.receiverRuntimeModelIds);
    NodeGraphHash::combinePodVector(hash, config.meshModelMatrices);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceBuffers);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceGradientBuffers);
    NodeGraphHash::combinePodVector(hash, config.receiverSurfaceGradientBufferOffsets);
    return hash;
}
