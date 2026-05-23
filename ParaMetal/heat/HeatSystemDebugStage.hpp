#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

#include "HeatSystemStageContext.hpp"

class MemoryAllocator;
class CommandPool;

class HeatSystemDebugStage {
public:
    explicit HeatSystemDebugStage(const HeatSystemStageContext& stageContext);

    void exportDebugArtifacts(bool debugEnable, uint32_t voronoiNodeCount,
        VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
        VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
        VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportDebugCellsToOBJ(bool debugEnable, uint32_t voronoiNodeCount,
        VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportCellVolumes(bool debugEnable, uint32_t voronoiNodeCount,
        VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportVoronoiDumpInfo(bool debugEnable, uint32_t voronoiNodeCount,
        VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
        VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);

private:
    HeatSystemStageContext context;
};
