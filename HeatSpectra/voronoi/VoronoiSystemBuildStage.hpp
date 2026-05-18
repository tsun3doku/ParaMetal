#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/VoronoiResources.hpp"

class MemoryAllocator;
class VoronoiGeoCompute;
class VoronoiModelRuntime;
class VulkanDevice;
class CommandPool;
class VoronoiSystemRuntime;

class VoronoiSystemBuildStage {
public:
    VoronoiSystemBuildStage(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, VoronoiResources& resources, CommandPool& renderCommandPool);
    ~VoronoiSystemBuildStage();

    bool buildVoronoiDiagram(VoronoiSystemRuntime& runtime, float cellSize, int voxelResolution, uint32_t maxNeighbors) const;
    bool dispatchVoronoiCompute(VoronoiSystemRuntime& runtime, bool debugEnable, uint32_t maxNeighbors);

    bool stageSurfaceMappings(VoronoiSystemRuntime& runtime) const;

    void setGhostFromVolumes(VoronoiSystemRuntime& runtime);
    void setGhostFromVoxelGrid(VoronoiSystemRuntime& runtime);

    void cleanupResources();

private:
    void initializeVoronoiGeoCompute();
    bool createGeometryBuffers(
        const std::vector<voronoi::Node>& initialNodes,
        const std::vector<glm::vec4>& seedPositions,
        const std::vector<uint32_t>& seedFlags,
        const std::vector<uint32_t>& neighborIndices,
        bool debugEnable,
        uint32_t maxNeighbors);
    bool buildGMLSInterfaceBuffer(VoronoiSystemRuntime& runtime, uint32_t maxNeighbors);
    bool rebuildOccupancyPointBuffer(VoronoiSystemRuntime& runtime) const;

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    VoronoiResources& resources;
    CommandPool& renderCommandPool;

    std::unique_ptr<VoronoiGeoCompute> voronoiGeoCompute;
};
