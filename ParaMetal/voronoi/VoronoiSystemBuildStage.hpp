#pragma once

#include "voronoi/VoronoiGpuStructs.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

class MemoryAllocator;
class VoronoiGeoCompute;
class VoronoiModelRuntime;
class VulkanDevice;
class CommandPool;
class VoronoiSystemRuntime;
class VoronoiNodeDomain;

class VoronoiSystemBuildStage {
  public:
    VoronoiSystemBuildStage(VulkanDevice &vulkanDevice, MemoryAllocator &memoryAllocator, CommandPool &commandPool);
    ~VoronoiSystemBuildStage();

    bool buildVoronoiDiagram(VoronoiSystemRuntime &runtime, float cellSize, int voxelResolution,
                             uint32_t maxNeighbors) const;
    bool dispatchVoronoiCompute(VoronoiSystemRuntime &runtime, bool debugEnabled, uint32_t maxNeighbors);

    bool stageSurfaceMappings(VoronoiSystemRuntime &runtime) const;

    void setGhostFromVolumes(VoronoiSystemRuntime &runtime);
    void setGhostFromVoxelGrid(VoronoiSystemRuntime &runtime);
    bool readSurfaceData(VoronoiSystemRuntime &runtime);

    void cleanupResources();
    void cleanup();

    uint32_t getCandidateNodeCount() const { return candidateNodeCount; }
    VkBuffer getCandidateNodeBuffer() const { return candidateNodeBuffer; }
    VkDeviceSize getCandidateNodeBufferOffset() const { return candidateNodeBufferOffset; }
    VkBuffer getCandidateNeighborIndicesBuffer() const { return candidateNeighborIndicesBuffer; }
    VkDeviceSize getCandidateNeighborIndicesBufferOffset() const { return candidateNeighborIndicesBufferOffset; }
    VkBuffer getSeedPositionBuffer() const { return seedPositionBuffer; }
    VkDeviceSize getSeedPositionBufferOffset() const { return seedPositionBufferOffset; }
    VkBuffer getNodeBuffer() const { return nodeBuffer; }
    VkDeviceSize getNodeBufferOffset() const { return nodeBufferOffset; }
    VkBuffer getCouplingBuffer() const { return couplingBuffer; }
    VkDeviceSize getCouplingBufferOffset() const { return couplingBufferOffset; }
    uint32_t getCouplingCount() const { return couplingCount; }
    VkBuffer getOccupancyPointBuffer() const { return occupancyPointBuffer; }
    VkDeviceSize getOccupancyPointBufferOffset() const { return occupancyPointBufferOffset; }
    uint32_t getOccupancyPointCount() const { return occupancyPointCount; }

  private:
    void initializeVoronoiGeoCompute();
    bool createCandidateBuffers(const std::vector<voronoi::Node> &candidateNodes,
                                const std::vector<glm::vec4> &candidatePositions,
                                const std::vector<uint32_t> &candidateNeighborIndices);
    bool createMeshGeometryBuffers(const std::vector<uint32_t> &nodeFlags, bool debugEnabled, uint32_t maxNeighbors);
    bool buildPointTopology(VoronoiSystemRuntime &runtime, std::vector<voronoi::Node> &candidateNodes,
                            const std::vector<glm::vec4> &candidatePositions,
                            const std::vector<uint32_t> &candidateNeighborIndices, uint32_t maxNeighbors);
    bool buildMeshTopology(VoronoiSystemRuntime &runtime, const std::vector<uint32_t> &candidateFlags,
                           bool debugEnabled, uint32_t maxNeighbors);
    bool buildMeshCouplingBuffer(VoronoiSystemRuntime &runtime, uint32_t maxNeighbors);
    bool finalizeNodeDomain(VoronoiSystemRuntime &runtime);
    bool uploadNodeDomainBuffers(const VoronoiNodeDomain &nodeDomain);
    void cleanupCandidateTopologyBuffers();
    void cleanupMeshGeometryBuffers();
    bool rebuildOccupancyPointBuffer(VoronoiSystemRuntime &runtime);

    VulkanDevice &vulkanDevice;
    MemoryAllocator &memoryAllocator;
    CommandPool &commandPool;

    uint32_t candidateNodeCount = 0;
    VkBuffer candidateNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateNodeBufferOffset = 0;
    VkBuffer candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateNeighborIndicesBufferOffset = 0;
    VkBuffer candidateInterfaceAreasBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateInterfaceAreasBufferOffset = 0;
    VkBuffer candidateInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateInterfaceNeighborIdsBufferOffset = 0;
    VkBuffer candidateCouplingBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateCouplingBufferOffset = 0;
    VkBuffer meshTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize meshTriangleBufferOffset = 0;
    VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedPositionBufferOffset = 0;
    VkBuffer nodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize nodeBufferOffset = 0;
    VkBuffer couplingBuffer = VK_NULL_HANDLE;
    VkDeviceSize couplingBufferOffset = 0;
    uint32_t couplingCount = 0;
    VkBuffer nodeFlagsBuffer = VK_NULL_HANDLE;
    VkDeviceSize nodeFlagsBufferOffset = 0;
    VkBuffer surfacePatchAreasBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfacePatchAreasBufferOffset = 0;
    std::vector<float> candidateSurfacePatchAreas;
    VkBuffer occupancyPointBuffer = VK_NULL_HANDLE;
    VkDeviceSize occupancyPointBufferOffset = 0;
    uint32_t occupancyPointCount = 0;
    VkBuffer voxelGridParamsBuffer = VK_NULL_HANDLE;
    VkDeviceSize voxelGridParamsBufferOffset = 0;
    VkBuffer voxelOccupancyBuffer = VK_NULL_HANDLE;
    VkDeviceSize voxelOccupancyBufferOffset = 0;
    VkBuffer voxelTrianglesListBuffer = VK_NULL_HANDLE;
    VkDeviceSize voxelTrianglesListBufferOffset = 0;
    VkBuffer voxelOffsetsBuffer = VK_NULL_HANDLE;
    VkDeviceSize voxelOffsetsBufferOffset = 0;
    VkBuffer debugCellGeometryBuffer = VK_NULL_HANDLE;
    VkDeviceSize debugCellGeometryBufferOffset = 0;
    VkBuffer voronoiDumpBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiDumpBufferOffset = 0;

    std::unique_ptr<VoronoiGeoCompute> voronoiGeoCompute;
};
