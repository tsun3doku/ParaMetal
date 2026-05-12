#include "HeatSystemDebugStage.hpp"

#include "HeatSystem.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/CommandBufferManager.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>

HeatSystemDebugStage::HeatSystemDebugStage(const HeatSystemStageContext& stageContext)
    : context(stageContext) {
}

void HeatSystemDebugStage::exportDebugArtifacts(bool debugEnable, uint32_t voronoiNodeCount,
    VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
    VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
    VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
    MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool) {
    if (!debugEnable) return;

    exportDebugCellsToOBJ(debugEnable, voronoiNodeCount,
        debugCellGeometryBuffer, debugCellGeometryBufferOffset, memoryAllocator, renderCommandPool);
    exportCellVolumes(debugEnable, voronoiNodeCount,
        voronoiNodeBuffer, voronoiNodeBufferOffset, memoryAllocator, renderCommandPool);
    exportVoronoiDumpInfo(debugEnable, voronoiNodeCount,
        voronoiNodeBuffer, voronoiNodeBufferOffset,
        voronoiDumpBuffer, voronoiDumpBufferOffset, memoryAllocator, renderCommandPool);
}

void HeatSystemDebugStage::exportDebugCellsToOBJ(bool debugEnable, uint32_t voronoiNodeCount,
    VkBuffer debugCellGeometryBuffer, VkDeviceSize debugCellGeometryBufferOffset,
    MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool) {
    if (!debugEnable) {
        return;
    }

    if (!debugCellGeometryBuffer) {
        return;
    }

    VkDeviceSize bufferSize = sizeof(voronoi::DebugCellGeometry) * voronoiNodeCount;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceSize stagingOff = 0;
    void* stagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, bufferSize, stagingBuf, stagingOff, &stagingMapped) != VK_SUCCESS || !stagingMapped) {
        return;
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    VkBufferCopy region{debugCellGeometryBufferOffset, stagingOff, bufferSize};
    vkCmdCopyBuffer(cmd, debugCellGeometryBuffer, stagingBuf, 1, &region);
    renderCommandPool.endCommands(cmd);

    voronoi::DebugCellGeometry* cells = static_cast<voronoi::DebugCellGeometry*>(stagingMapped);

    std::ofstream obj("voronoi_unrestricted_debug_cells.obj");
    if (!obj) {
        std::cerr << "[HeatSystem] Failed to create OBJ file!" << std::endl;
        return;
    }

    char buffer[65536];
    obj.rdbuf()->pubsetbuf(buffer, sizeof(buffer));

    obj << "# Unrestricted Voronoi Cells\n";
    obj << "o Voronoi_Cells_Combined\n";
    uint32_t offset = 1;
    uint32_t exportCount = 0;

    for (uint32_t i = 0; i < voronoiNodeCount; i++) {
        if (cells[i].vertexCount == 0) {
            continue;
        }

        for (uint32_t v = 0; v < cells[i].vertexCount; v++) {
            obj << "v " << cells[i].vertices[v].x << " " << cells[i].vertices[v].y << " " << cells[i].vertices[v].z << "\n";
        }
        for (uint32_t t = 0; t < cells[i].triangleCount; t++) {
            obj << "f " << (offset + cells[i].triangles[t].x) << " "
                << (offset + cells[i].triangles[t].y) << " "
                << (offset + cells[i].triangles[t].z) << "\n";
        }
        offset += cells[i].vertexCount;
        exportCount++;
    }

    obj.close();
    memoryAllocator.free(stagingBuf, stagingOff);
}

void HeatSystemDebugStage::exportCellVolumes(bool debugEnable, uint32_t voronoiNodeCount,
    VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
    MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool) {
    if (!debugEnable) {
        return;
    }

    if (!voronoiNodeBuffer) {
        std::cerr << "[HeatSystem] Error: voronoi::Node buffer not available" << std::endl;
        return;
    }

    VkDeviceSize bufferSize = sizeof(voronoi::Node) * voronoiNodeCount;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VkDeviceSize stagingOff = 0;
    void* stagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, bufferSize, stagingBuf, stagingOff, &stagingMapped) != VK_SUCCESS || !stagingMapped) {
        return;
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    VkBufferCopy region{voronoiNodeBufferOffset, stagingOff, bufferSize};
    vkCmdCopyBuffer(cmd, voronoiNodeBuffer, stagingBuf, 1, &region);
    renderCommandPool.endCommands(cmd);

    voronoi::Node* nodes = static_cast<voronoi::Node*>(stagingMapped);

    std::ofstream volumeFile("cell_volumes.txt");
    volumeFile << "# Cell Index -> Restricted Volume\n";
    volumeFile << "# NOTE: Using GPU filtered seed positions (cells.size() = " << voronoiNodeCount << ")\n";
    for (uint32_t i = 0; i < voronoiNodeCount; i++) {
        volumeFile << "Cell " << i << " -> Volume: " << nodes[i].volume << "\n";
    }

    volumeFile.close();
    memoryAllocator.free(stagingBuf, stagingOff);
}

void HeatSystemDebugStage::exportVoronoiDumpInfo(bool debugEnable, uint32_t voronoiNodeCount,
    VkBuffer voronoiNodeBuffer, VkDeviceSize voronoiNodeBufferOffset,
    VkBuffer voronoiDumpBuffer, VkDeviceSize voronoiDumpBufferOffset,
    MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool) {
    if (!debugEnable) {
        return;
    }

    if (!voronoiDumpBuffer) {
        std::cerr << "[HeatSystem] Error: voronoi::DumpInfo buffer not available" << std::endl;
        return;
    }

    VkDeviceSize dumpBufferSize = sizeof(voronoi::DumpInfo) * voronoi::DEBUG_DUMP_CELL_COUNT;
    VkBuffer dumpStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize dumpStagingOff = 0;
    void* dumpStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, dumpBufferSize, dumpStagingBuf, dumpStagingOff, &dumpStagingMapped) != VK_SUCCESS || !dumpStagingMapped) {
        return;
    }

    VkDeviceSize nodeBufferSize = sizeof(voronoi::Node) * voronoiNodeCount;
    VkBuffer nodeStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize nodeStagingOff = 0;
    void* nodeStagingMapped = nullptr;
    if (voronoiNodeBuffer && createStagingBuffer(memoryAllocator, nodeBufferSize, nodeStagingBuf, nodeStagingOff, &nodeStagingMapped) != VK_SUCCESS) {
        memoryAllocator.free(dumpStagingBuf, dumpStagingOff);
        return;
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    VkBufferCopy dumpRegion{voronoiDumpBufferOffset, dumpStagingOff, dumpBufferSize};
    vkCmdCopyBuffer(cmd, voronoiDumpBuffer, dumpStagingBuf, 1, &dumpRegion);
    if (voronoiNodeBuffer && nodeStagingBuf) {
        VkBufferCopy nodeRegion{voronoiNodeBufferOffset, nodeStagingOff, nodeBufferSize};
        vkCmdCopyBuffer(cmd, voronoiNodeBuffer, nodeStagingBuf, 1, &nodeRegion);
    }
    renderCommandPool.endCommands(cmd);

    voronoi::DumpInfo* dumpInfos = static_cast<voronoi::DumpInfo*>(dumpStagingMapped);

    double totalRestrictedVolumePos = 0.0;
    double totalRestrictedVolumeNegAbs = 0.0;
    uint32_t negativeVolumeCount = 0;
    if (nodeStagingMapped) {
        const voronoi::Node* nodes = static_cast<const voronoi::Node*>(nodeStagingMapped);
        for (uint32_t i = 0; i < voronoiNodeCount; i++) {
            const double v = (double)nodes[i].volume;
            if (v > 0.0) {
                totalRestrictedVolumePos += v;
            } else if (v < 0.0) {
                negativeVolumeCount++;
                totalRestrictedVolumeNegAbs += -v;
            }
        }
    }

    for (uint32_t slot = 0; slot < voronoi::DEBUG_DUMP_CELL_COUNT; slot++) {
        dumpInfos[slot].totalMeshVolume = static_cast<float>(totalRestrictedVolumePos);
        dumpInfos[slot].negativeVolumeCellCount = negativeVolumeCount;
        dumpInfos[slot].negativeVolumeSumAbs = static_cast<float>(totalRestrictedVolumeNegAbs);
    }

    std::ofstream dumpFile("voronoi_debug_dump.txt");
    dumpFile << std::scientific << std::setprecision(10);

    dumpFile << "Total Mesh Volume: " << totalRestrictedVolumePos << "\n";
    dumpFile << "Negative volume cells: " << negativeVolumeCount << " Sum =" << totalRestrictedVolumeNegAbs << ")\n";
    dumpFile << "\n";

    for (uint32_t slot = 0; slot < voronoi::DEBUG_DUMP_CELL_COUNT; slot++) {
        const voronoi::DumpInfo& info = dumpInfos[slot];

        if (info.cellID == 0) {
            continue;
        }

        dumpFile << "CELL " << info.cellID << " (Slot " << slot << ")\n";
        dumpFile << "Seed: (" << info.seedPos.x << ", " << info.seedPos.y << ", " << info.seedPos.z << ")\n";
        dumpFile << "Volumes:\n";
        dumpFile << "  Unrestricted: " << info.unrestrictedVolume << "\n";
        dumpFile << "  Restricted:   " << info.restrictedVolume << "\n";
        dumpFile << "Plane Areas (" << info.planeAreaCount << "):\n";
        for (uint32_t i = 0; i < info.planeAreaCount && i < voronoi::DEBUG_MAX_PLANE_AREAS; ++i) {
            const voronoi::DebugPlaneArea& planeArea = info.planeAreas[i];
            dumpFile << "  Plane " << planeArea.planeIndex
                     << " -> Neighbor " << planeArea.neighborCellID
                     << ", Area: " << planeArea.area << "\n";
        }
        dumpFile << "\n";
    }

    dumpFile.close();
    memoryAllocator.free(dumpStagingBuf, dumpStagingOff);
    if (nodeStagingBuf) {
        memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
    }
}
