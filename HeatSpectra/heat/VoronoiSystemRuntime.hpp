#pragma once

#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "voronoi/VoronoiResources.hpp"
#include "voronoi/VoronoiSeeder.hpp"
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
    bool isSeederReady() const { return voronoiSeederReady; }

    const std::vector<std::unique_ptr<VoronoiModelRuntime>>& getModelRuntimes() const { return modelRuntimes; }
    VoronoiModelRuntime* getModelRuntime() const { return modelRuntimes.empty() ? nullptr : modelRuntimes.front().get(); }

    uint32_t getVoronoiNodeCount() const { return resources.voronoiNodeCount; }

    VoronoiResources& resourcesRef() { return resources; }
    const VoronoiResources& resourcesRef() const { return resources; }

    // VoronoiDomain fields moved here
    VoronoiSeeder* getSeeder() const { return seeder.get(); }
    VoronoiIntegrator* getIntegrator() const { return integrator.get(); }
    void resetSeeder() { seeder = std::make_unique<VoronoiSeeder>(); }
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

    void setReceiverGeometry(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
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
    float getCellSize() const { return cellSize; }
    int getVoxelResolution() const { return voxelResolution; }
    void markSeederReady();
    void markReady();
    void uploadModelStagingBuffers(CommandPool& renderCommandPool);

    void cleanupResources(VulkanDevice& vulkanDevice);
    void cleanup(MemoryAllocator& memoryAllocator);

private:
    void invalidateMaterialization();

    std::vector<std::unique_ptr<VoronoiModelRuntime>> modelRuntimes;
    float cellSize = 0.005f;
    int voxelResolution = 128;

    // VoronoiDomain fields moved here
    std::unique_ptr<VoronoiSeeder> seeder;
    std::unique_ptr<VoronoiIntegrator> integrator;
    VoxelGrid voxelGrid;
    bool voxelGridBuilt = false;
    std::vector<uint32_t> seedFlags;
    std::vector<glm::vec4> seedPositions;
    std::vector<uint32_t> neighborIndices;
    std::vector<std::array<glm::vec4, 3>> meshTriangles;

    VoronoiResources resources;
    bool voronoiSeederReady = false;
    bool voronoiReady = false;
    static constexpr float AMBIENT_TEMPERATURE = 1.0f;
};
