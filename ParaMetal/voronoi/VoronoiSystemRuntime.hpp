#pragma once

#include "voronoi/VoronoiDomainRuntime.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "voronoi/VoronoiMeshGrid.hpp"
#include "voronoi/VoronoiIntegrator.hpp"
#include "voronoi/VoronoiNodeDomain.hpp"
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

    uint32_t getNodeCount() const { return nodeDomain.getNodeCount(); }
    VoronoiNodeDomain& getNodeDomain() { return nodeDomain; }
    const VoronoiNodeDomain& getNodeDomain() const { return nodeDomain; }

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
    void setMeshGeometry(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        const std::vector<glm::vec3>& geometryPositions,
        const std::vector<uint32_t>& geometryTriangleIndices,
        const std::vector<voronoi::SurfaceVertex>& surfaceVertices,
        const std::vector<uint32_t>& surfaceTriangleIndices,
        uint32_t runtimeModelId,
        const glm::mat4& meshModelMatrix);
    void setPointGeometry(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        uint64_t domainKey,
        const std::vector<glm::vec4>& positions);
    void clearGeometry();
    void setParams(float cellSize, int voxelResolution);
    float getCellSize() const { return cellSize; }
    int getVoxelResolution() const { return voxelResolution; }
    void markMeshGridReady();
    void markReady();

    void setSeedPositions(const std::vector<glm::vec4>& positions);

    void cleanup(MemoryAllocator& memoryAllocator);

private:
    void invalidateMaterialization();

    std::unique_ptr<VoronoiDomainRuntime> domainRuntime;
    float cellSize = 0.005f;
    int voxelResolution = 128;

    std::unique_ptr<VoronoiMeshGrid> meshGrid;
    std::unique_ptr<VoronoiIntegrator> integrator;
    VoxelGrid voxelGrid;
    bool voxelGridBuilt = false;
    std::vector<uint32_t> seedFlags;
    std::vector<glm::vec4> seedPositions;
    std::vector<uint32_t> neighborIndices;
    std::vector<std::array<glm::vec4, 3>> meshTriangles;
    VoronoiNodeDomain nodeDomain;

    bool voronoiMeshGridReady = false;
    bool voronoiReady = false;
};
