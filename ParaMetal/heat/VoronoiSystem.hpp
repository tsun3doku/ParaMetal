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
        CommandPool& commandPool);
    ~VoronoiSystem();

    bool isInitialized() const { return initialized; }
    bool isReady() const { return runtime.isReady(); }

    void setReceiverGeometry(
        uint32_t receiverNodeModelId,
        const std::vector<glm::vec3>& receiverGeometryPositions,
        const std::vector<uint32_t>& receiverGeometryTriangleIndices,
        const SupportingHalfedge::IntrinsicMesh& receiverIntrinsicMesh,
        const std::vector<VoronoiModelRuntime::SurfaceVertex>& receiverSurfaceVertices,
        const std::vector<uint32_t>& receiverIntrinsicTriangleIndices,
        uint32_t receiverModelId,
        const glm::mat4& meshModelMatrix);
    void setPointGeometry(const std::vector<glm::vec4>& positions);
    void setSeedPositions(const std::vector<glm::vec4>& positions);
    void clearReceiverGeometry();
    void setParams(float cellSize, int voxelResolution);
    bool ensureConfigured();

    VoronoiDomainRuntime* getDomainRuntime() const { return runtime.getDomainRuntime(); }
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
    CommandPool& commandPool;
    VoronoiSystemRuntime runtime;

    std::unique_ptr<VoronoiSystemBuildStage> voronoiSystemBuildStage;
    std::unique_ptr<VoronoiCandidateCompute> voronoiCandidateCompute;

    uint32_t maxFramesInFlight;
    bool initialized = false;
    bool debugEnable = false;
    static constexpr int K_NEIGHBORS = 50;
};
