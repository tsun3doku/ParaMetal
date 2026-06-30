#pragma once

#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiDomainRuntime.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "voronoi/VoronoiResources.hpp"
#include "voronoi/VoronoiMeshGrid.hpp"
#include "voronoi/VoronoiIntegrator.hpp"
#include "spatial/SpatialOrder.hpp"
#include "spatial/VoxelGrid.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

class CommandPool;
class MemoryAllocator;
class ModelRegistry;
class VulkanDevice;
class VoronoiCandidateCompute;
class VoronoiGeoCompute;

class VoronoiSystemRuntime {
public:
    bool isReady() const { return voronoiReady; }
    bool isMeshGridReady() const { return voronoiMeshGridReady; }

    VoronoiDomainRuntime* getDomainRuntime() const { return domainRuntime.get(); }

    uint32_t getVoronoiNodeCount() const { return resources.voronoiNodeCount; }
    uint32_t getSimNodeCount() const { return simNodeCount; }
    VkBuffer getSimNodeBuffer() const { return simNodeBuffer; }
    VkDeviceSize getSimNodeBufferOffset() const { return simNodeBufferOffset; }
    VkDeviceSize getSimNodeBufferSize() const { return simNodeBufferSize; }
    VkBuffer getSimGMLSInterfaceBuffer() const { return simGMLSInterfaceBuffer; }
    VkDeviceSize getSimGMLSInterfaceBufferOffset() const { return simGMLSInterfaceBufferOffset; }
    uint32_t getSimGMLSInterfaceCount() const { return simGMLSInterfaceCount; }
    const std::vector<uint32_t>& getVoronoiToSim() const { return voronoiToSim; }
    const std::vector<uint32_t>& getSimToVoronoi() const { return simToVoronoi; }
    const std::vector<float>& getSimNodeVolumes() const { return simNodeVolumes; }

    VoronoiResources& resourcesRef() { return resources; }
    const VoronoiResources& resourcesRef() const { return resources; }

    // VoronoiDomain fields moved here
    VoronoiMeshGrid* getMeshGrid() const { return meshGrid.get(); }
    VoronoiIntegrator* getIntegrator() const { return integrator.get(); }
    void resetMeshGrid() { meshGrid = std::make_unique<VoronoiMeshGrid>(); }
    void resetIntegrator() { integrator = std::make_unique<VoronoiIntegrator>(); }
    VoxelGrid& getVoxelGrid() { return voxelGrid; }
    const VoxelGrid& getVoxelGrid() const { return voxelGrid; }
    bool isVoxelGridBuilt() const { return voxelGridBuilt; }
    void setVoxelGridBuilt(bool built) { voxelGridBuilt = built; }
    std::vector<uint32_t>& getSeedFlags() { return seedFlags; }
    const std::vector<uint32_t>& getSeedFlags() const { return seedFlags; }

    std::vector<glm::vec4>& getSeedPositions() { return seedPositions; }
    const std::vector<glm::vec4>& getSeedPositions() const { return seedPositions; }
    std::vector<uint32_t>& getNeighborIndices() { return neighborIndices; }
    const std::vector<uint32_t>& getNeighborIndices() const { return neighborIndices; }
    std::vector<std::array<glm::vec4, 3>>& getMeshTriangles() { return meshTriangles; }
    const std::vector<std::array<glm::vec4, 3>>& getMeshTriangles() const { return meshTriangles; }

    void reorderNodes();
    void buildSimSpaceMapping();
    bool buildSimBuffers(MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);

    void setReceiverGeometry(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        uint32_t receiverNodeModelId,
        const std::vector<glm::vec3>& receiverGeometryPositions,
        const std::vector<uint32_t>& receiverGeometryTriangleIndices,
        const SupportingHalfedge::IntrinsicMesh& receiverIntrinsicMesh,
        const std::vector<VoronoiModelRuntime::SurfaceVertex>& receiverSurfaceVertices,
        const std::vector<uint32_t>& receiverIntrinsicTriangleIndices,
        uint32_t receiverModelId,
        const glm::mat4& meshModelMatrix);
    void setPointGeometry(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        uint64_t domainKey,
        const std::vector<glm::vec4>& positions);
    void clearReceiverGeometry();
    void setParams(float cellSize, int voxelResolution);
    float getCellSize() const { return cellSize; }
    int getVoxelResolution() const { return voxelResolution; }
    void markMeshGridReady();
    void markReady();

    void setSeedPositions(const std::vector<glm::vec4>& positions);

    void cleanupResources(VulkanDevice& vulkanDevice);
    void cleanup(MemoryAllocator& memoryAllocator);

private:
    void invalidateMaterialization();

    std::unique_ptr<VoronoiDomainRuntime> domainRuntime;
    float cellSize = 0.005f;
    int voxelResolution = 128;

    // VoronoiDomain fields moved here
    std::unique_ptr<VoronoiMeshGrid> meshGrid;
    std::unique_ptr<VoronoiIntegrator> integrator;
    VoxelGrid voxelGrid;
    bool voxelGridBuilt = false;
    std::vector<uint32_t> seedFlags;
    std::vector<glm::vec4> seedPositions;
    std::vector<uint32_t> neighborIndices;
    std::vector<std::array<glm::vec4, 3>> meshTriangles;
    std::vector<uint32_t> voronoiToSim;
    std::vector<uint32_t> simToVoronoi;
    std::vector<float> simNodeVolumes;
    uint32_t simNodeCount = 0;
    VkBuffer simNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize simNodeBufferOffset = 0;
    VkDeviceSize simNodeBufferSize = 0;
    VkBuffer simGMLSInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize simGMLSInterfaceBufferOffset = 0;
    uint32_t simGMLSInterfaceCount = 0;

    VoronoiResources resources;
    bool voronoiMeshGridReady = false;
    bool voronoiReady = false;
    static constexpr float AMBIENT_TEMPERATURE = 1.0f;
};
