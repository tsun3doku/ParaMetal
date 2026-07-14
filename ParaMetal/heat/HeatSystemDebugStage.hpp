#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

class MemoryAllocator;
class CommandPool;

class HeatSystemDebugStage {
public:
    explicit HeatSystemDebugStage() = default;

    void exportDebugArtifacts(bool debugEnable, uint32_t candidateNodeCount,
        VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
        VkBuffer candidateNodeBuffer, VkDeviceSize candidateNodeBufferOffset,
        VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportDebugCellsToOBJ(bool debugEnable, uint32_t candidateNodeCount,
        VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportCellVolumes(bool debugEnable, uint32_t candidateNodeCount,
        VkBuffer candidateNodeBuffer, VkDeviceSize candidateNodeBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
    void exportVoronoiDumpInfo(bool debugEnable, uint32_t candidateNodeCount,
        VkBuffer candidateNodeBuffer, VkDeviceSize candidateNodeBufferOffset,
        VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
        MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool);
};
