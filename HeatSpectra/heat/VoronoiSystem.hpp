#pragma once

#include "VoronoiSystemRuntime.hpp"
#include "voronoi/VoronoiSystemBuildStage.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class VoronoiModelRuntime;
class MemoryAllocator;
class ModelRegistry;
class VulkanDevice;
class CommandPool;
class VoronoiCandidateCompute;

class VoronoiSystem {
public:
    VoronoiSystem(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        ModelRegistry& resourceManager,
        uint32_t maxFramesInFlight,
        CommandPool& renderCommandPool);
    ~VoronoiSystem();

    bool isInitialized() const { return initialized; }
    bool isReady() const { return runtime.isReady(); }

    void setReceiverGeometry(
        const std::vector<uint32_t>& receiverNodeModelIds,
        const std::vector<std::vector<glm::vec3>>& receiverGeometryPositions,
        const std::vector<std::vector<uint32_t>>& receiverGeometryTriangleIndices,
        const std::vector<SupportingHalfedge::IntrinsicMesh>& receiverIntrinsicMeshes,
        const std::vector<std::vector<VoronoiModelRuntime::SurfaceVertex>>& receiverSurfaceVertices,
        const std::vector<std::vector<uint32_t>>& receiverIntrinsicTriangleIndices,
        const std::vector<uint32_t>& receiverModelIds,
        const std::vector<VkBuffer>& meshVertexBuffers,
        const std::vector<VkDeviceSize>& meshVertexBufferOffsets,
        const std::vector<VkBuffer>& meshIndexBuffers,
        const std::vector<VkDeviceSize>& meshIndexBufferOffsets,
        const std::vector<uint32_t>& meshIndexCounts,
        const std::vector<glm::mat4>& meshModelMatrices,
        const std::vector<VkBufferView>& supportingHalfedgeViews,
        const std::vector<VkBufferView>& supportingAngleViews,
        const std::vector<VkBufferView>& halfedgeViews,
        const std::vector<VkBufferView>& edgeViews,
        const std::vector<VkBufferView>& triangleViews,
        const std::vector<VkBufferView>& lengthViews,
        const std::vector<VkBufferView>& inputHalfedgeViews,
        const std::vector<VkBufferView>& inputEdgeViews,
        const std::vector<VkBufferView>& inputTriangleViews,
        const std::vector<VkBufferView>& inputLengthViews);
    void clearReceiverGeometry();
    void setParams(float cellSize, int voxelResolution);
    bool ensureConfigured();

    const std::vector<std::unique_ptr<VoronoiModelRuntime>>& getModelRuntimes() const { return runtime.getModelRuntimes(); }
    VoronoiModelRuntime* getModelRuntime() const { return runtime.getModelRuntime(); }
    uint32_t getVoronoiNodeCount() const { return runtime.getVoronoiNodeCount(); }

    VoronoiResources& resourcesRef() { return runtime.resourcesRef(); }
    const VoronoiResources& resourcesRef() const { return runtime.resourcesRef(); }
    VoronoiSystemRuntime& runtimeRef() { return runtime; }
    const VoronoiSystemRuntime& runtimeRef() const { return runtime; }

    void cleanupResources();
    void cleanup();

private:
    void failInitialization(const char* stage);
    void initializeVoronoiCandidateCompute();
    bool rebuildVoronoiRuntime();
    void executeBufferTransfers();
    void dispatchVoronoiCandidateUpdates();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& renderCommandPool;
    VoronoiSystemRuntime runtime;

    std::unique_ptr<VoronoiSystemBuildStage> voronoiSystemBuildStage;
    std::unique_ptr<VoronoiCandidateCompute> voronoiCandidateCompute;

    uint32_t maxFramesInFlight;
    bool initialized = false;
    bool debugEnable = false;
    static constexpr int K_NEIGHBORS = 50;
};
