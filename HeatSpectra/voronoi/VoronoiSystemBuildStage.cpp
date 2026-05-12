#include "VoronoiSystemBuildStage.hpp"

#include "spatial/VoxelGrid.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "heat/VoronoiSystemRuntime.hpp"

#include "renderers/PointRenderer.hpp"
#include "util/GMLS.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiGeoCompute.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <libs/nanoflann/include/nanoflann.hpp>
#include "voronoi/VoronoiAdapters.hpp"


VoronoiSystemBuildStage::VoronoiSystemBuildStage(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    VoronoiResources& resources,
    CommandPool& renderCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resources(resources),
      renderCommandPool(renderCommandPool) {
    initializeVoronoiGeoCompute();
}

VoronoiSystemBuildStage::~VoronoiSystemBuildStage() {
}

void VoronoiSystemBuildStage::initializeVoronoiGeoCompute() {
    voronoiGeoCompute = std::make_unique<VoronoiGeoCompute>(vulkanDevice, renderCommandPool);
}

bool VoronoiSystemBuildStage::buildVoronoiDiagram(
    VoronoiSystemRuntime& runtime,
    float cellSize,
    int voxelResolution,
    uint32_t maxNeighbors) const {
    VoronoiModelRuntime* modelRuntime = runtime.getModelRuntime();
    if (!modelRuntime) {
        return false;
    }

    const uint32_t receiverModelId = modelRuntime->getRuntimeModelId();
    if (receiverModelId == 0) {
        return false;
    }

    runtime.resetSeeder();
    runtime.resetIntegrator();
    if (!runtime.getSeeder() || !runtime.getIntegrator()) {
        return false;
    }

    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh = modelRuntime->getIntrinsicMesh();
    const std::vector<glm::vec3>& geometryPositions = modelRuntime->getGeometryPositions();
    const std::vector<uint32_t>& geometryIndices = modelRuntime->getGeometryTriangleIndices();
    runtime.getSeeder()->generateSeeds(
        intrinsicMesh,
        geometryPositions,
        geometryIndices,
        cellSize,
        runtime.getVoxelGrid(),
        voxelResolution);
    runtime.setVoxelGridBuilt(true);

    const std::vector<VoronoiSeeder::Seed>& seeds = runtime.getSeeder()->getSeeds();
    if (seeds.empty()) {
        return false;
    }

    std::vector<glm::dvec3> seedPositions;
    seedPositions.reserve(seeds.size());
    runtime.getSeedFlags().clear();
    runtime.getSeedFlags().reserve(seeds.size());
    for (const VoronoiSeeder::Seed& seed : seeds) {
        seedPositions.push_back(glm::dvec3(seed.pos));
        uint32_t flags = 0u;
        if (seed.isSurface) {
            flags |= 2u;
        }
        runtime.getSeedFlags().push_back(flags);
    }

    runtime.getIntegrator()->computeNeighbors(seedPositions, maxNeighbors, runtime.getSeedPositions(), runtime.getNeighborIndices());
    runtime.getIntegrator()->extractMeshTriangles(geometryPositions, geometryIndices, runtime.getMeshTriangles());

    return true;
}

void VoronoiSystemBuildStage::setGhostFromVolumes(VoronoiSystemRuntime& runtime) {
    if (resources.voronoiNodeCount == 0 || !resources.seedFlagsBuffer) {
        return;
    }

    VkDeviceSize nodeBufferSize = sizeof(voronoi::Node) * resources.voronoiNodeCount;
    VkBuffer nodeStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize nodeStagingOff = 0;
    void* nodeStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, nodeBufferSize, nodeStagingBuf, nodeStagingOff, &nodeStagingMapped) != VK_SUCCESS || !nodeStagingMapped) {
        return;
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    VkBufferCopy nodeRegion{resources.voronoiNodeBufferOffset, nodeStagingOff, nodeBufferSize};
    vkCmdCopyBuffer(cmd, resources.voronoiNodeBuffer, nodeStagingBuf, 1, &nodeRegion);
    renderCommandPool.endCommands(cmd);

    const voronoi::Node* nodes = static_cast<const voronoi::Node*>(nodeStagingMapped);

    uint32_t promotedCount = 0;
    for (uint32_t i = 0; i < resources.voronoiNodeCount; ++i) {
        if ((runtime.getSeedFlags()[i] & 1u) != 0u) {
            continue;
        }
        if (std::abs(nodes[i].volume) <= 1e-12f) {
            runtime.getSeedFlags()[i] |= 1u;
            ++promotedCount;
        }
    }

    memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
}

void VoronoiSystemBuildStage::setGhostFromVoxelGrid(VoronoiSystemRuntime& runtime) {
    if (!runtime.isVoxelGridBuilt() || !runtime.getSeeder()) {
        return;
    }
    const auto& seedPositions = runtime.getSeedPositions();
    if (seedPositions.size() != runtime.getSeedFlags().size()) {
        return;
    }
    const float maxDistFromSurface = runtime.getSeeder()->getCellSize() * 1.5f;
    uint32_t ghostCount = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(seedPositions.size()); ++i) {
        const glm::vec4& seed = seedPositions[i];
        glm::ivec3 voxel = runtime.getVoxelGrid().worldToVoxel(glm::vec3(seed));
        uint8_t occ = runtime.getVoxelGrid().getOccupancy(voxel.x, voxel.y, voxel.z);
        bool isInside = (occ == 2 || occ == 1);
        float distToSurface = runtime.getSeeder()->sampleSDFGrid(glm::vec3(seed));
        if (!isInside && distToSurface > maxDistFromSurface) {
            runtime.getSeedFlags()[i] |= 1u;
            ++ghostCount;
        }
    }
}

bool VoronoiSystemBuildStage::createGeometryBuffers(
    const std::vector<voronoi::Node>& initialNodes,
    const std::vector<glm::vec4>& seedPositions,
    const std::vector<uint32_t>& seedFlags,
    const std::vector<uint32_t>& neighborIndices,
    bool debugEnable,
    uint32_t maxNeighbors) {
    if (seedPositions.empty() || seedFlags.empty() || initialNodes.empty()) {
        return false;
    }

    if (seedPositions.size() != seedFlags.size() || seedPositions.size() != initialNodes.size()) {
        return false;
    }

    resources.voronoiNodeCount = static_cast<uint32_t>(seedPositions.size());

    VkDeviceSize bufferSize = sizeof(voronoi::Node) * resources.voronoiNodeCount;
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, initialNodes.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.voronoiNodeBuffer = buf;
        resources.voronoiNodeBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.voronoiNodeBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    std::vector<uint32_t> flattenedNeighbors = neighborIndices;
    if (flattenedNeighbors.empty()) {
        flattenedNeighbors.resize(
            static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors),
            UINT32_MAX);
    }

    bufferSize = sizeof(uint32_t) * flattenedNeighbors.size();
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, flattenedNeighbors.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.neighborIndicesBuffer = buf;
        resources.neighborIndicesBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.neighborIndicesBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    const size_t interfaceDataSize =
        static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors);
    std::vector<float> emptyAreas(interfaceDataSize, 0.0f);
    bufferSize = sizeof(float) * interfaceDataSize;

    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, emptyAreas.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.interfaceAreasBuffer = buf;
        resources.interfaceAreasBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.interfaceAreasBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    std::vector<uint32_t> emptyIds(interfaceDataSize, UINT32_MAX);
    bufferSize = sizeof(uint32_t) * interfaceDataSize;

    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, emptyIds.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.interfaceNeighborIdsBuffer = buf;
        resources.interfaceNeighborIdsBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.interfaceNeighborIdsBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    uint32_t numDebugCells = debugEnable ? resources.voronoiNodeCount : 1u;
    std::vector<voronoi::DebugCellGeometry> debugCells(numDebugCells);
    for (auto& cell : debugCells) {
        cell.cellID = 0;
        cell.vertexCount = 0;
        cell.triangleCount = 0;
        cell.volume = 0.0f;
    }

    bufferSize = sizeof(voronoi::DebugCellGeometry) * numDebugCells;
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, debugCells.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.debugCellGeometryBuffer = buf;
        resources.debugCellGeometryBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.debugCellGeometryBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    uint32_t dumpCount = debugEnable ? voronoi::DEBUG_DUMP_CELL_COUNT : 1u;
    std::vector<voronoi::DumpInfo> dumpInfos(dumpCount);
    std::memset(dumpInfos.data(), 0, sizeof(voronoi::DumpInfo) * dumpCount);

    bufferSize = sizeof(voronoi::DumpInfo) * dumpCount;
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, dumpInfos.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.voronoiDumpBuffer = buf;
        resources.voronoiDumpBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.voronoiDumpBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    bufferSize = sizeof(glm::vec4) * seedPositions.size();
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, seedPositions.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.seedPositionBuffer = buf;
        resources.seedPositionBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.seedPositionBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    bufferSize = sizeof(uint32_t) * seedFlags.size();
    {
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, seedFlags.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.seedFlagsBuffer = buf;
        resources.seedFlagsBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.seedFlagsBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    return true;
}

bool VoronoiSystemBuildStage::buildGMLSInterfaceBuffer(VoronoiSystemRuntime& runtime, uint32_t maxNeighbors) {
    if (resources.voronoiNodeCount == 0) {
        return false;
    }

    VkDeviceSize areasBufferSize = sizeof(float) * static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors);
    VkBuffer areasStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize areasStagingOff = 0;
    void* areasStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, areasBufferSize, areasStagingBuf, areasStagingOff, &areasStagingMapped) != VK_SUCCESS || !areasStagingMapped) {
        return false;
    }

    VkDeviceSize neighborIdsBufferSize = sizeof(uint32_t) * static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors);
    VkBuffer neighborIdsStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize neighborIdsStagingOff = 0;
    void* neighborIdsStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, neighborIdsBufferSize, neighborIdsStagingBuf, neighborIdsStagingOff, &neighborIdsStagingMapped) != VK_SUCCESS || !neighborIdsStagingMapped) {
        memoryAllocator.free(areasStagingBuf, areasStagingOff);
        return false;
    }

    VkDeviceSize nodeBufferSize = sizeof(voronoi::Node) * resources.voronoiNodeCount;
    VkBuffer nodeStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize nodeStagingOff = 0;
    void* nodeStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, nodeBufferSize, nodeStagingBuf, nodeStagingOff, &nodeStagingMapped) != VK_SUCCESS || !nodeStagingMapped) {
        memoryAllocator.free(areasStagingBuf, areasStagingOff);
        memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
        return false;
    }

    VkDeviceSize seedPosBufferSize = sizeof(glm::vec4) * resources.voronoiNodeCount;
    VkBuffer seedPosStagingBuf = VK_NULL_HANDLE;
    VkDeviceSize seedPosStagingOff = 0;
    void* seedPosStagingMapped = nullptr;
    if (createStagingBuffer(memoryAllocator, seedPosBufferSize, seedPosStagingBuf, seedPosStagingOff, &seedPosStagingMapped) != VK_SUCCESS || !seedPosStagingMapped) {
        memoryAllocator.free(areasStagingBuf, areasStagingOff);
        memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
        memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
        return false;
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    VkBufferCopy areasRegion{resources.interfaceAreasBufferOffset, areasStagingOff, areasBufferSize};
    vkCmdCopyBuffer(cmd, resources.interfaceAreasBuffer, areasStagingBuf, 1, &areasRegion);
    VkBufferCopy neighborIdsRegion{resources.interfaceNeighborIdsBufferOffset, neighborIdsStagingOff, neighborIdsBufferSize};
    vkCmdCopyBuffer(cmd, resources.interfaceNeighborIdsBuffer, neighborIdsStagingBuf, 1, &neighborIdsRegion);
    VkBufferCopy nodeRegion{resources.voronoiNodeBufferOffset, nodeStagingOff, nodeBufferSize};
    vkCmdCopyBuffer(cmd, resources.voronoiNodeBuffer, nodeStagingBuf, 1, &nodeRegion);
    VkBufferCopy seedPosRegion{resources.seedPositionBufferOffset, seedPosStagingOff, seedPosBufferSize};
    vkCmdCopyBuffer(cmd, resources.seedPositionBuffer, seedPosStagingBuf, 1, &seedPosRegion);
    renderCommandPool.endCommands(cmd);

    float* areas = static_cast<float*>(areasStagingMapped);
    uint32_t* neighborIds = static_cast<uint32_t*>(neighborIdsStagingMapped);
    voronoi::Node* nodes = static_cast<voronoi::Node*>(nodeStagingMapped);
    const glm::vec4* seedPositions = static_cast<const glm::vec4*>(seedPosStagingMapped);

    if (!areas || !neighborIds || !nodes || !seedPositions) {
        memoryAllocator.free(areasStagingBuf, areasStagingOff);
        memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
        memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
        memoryAllocator.free(seedPosStagingBuf, seedPosStagingOff);
        return false;
    }

    const std::vector<uint32_t>& seedFlags = runtime.getSeedFlags();

    freeBuffer(resources.gmlsInterfaceBuffer, resources.gmlsInterfaceBufferOffset);

    std::vector<voronoi::GMLSInterface> interfaces;
    interfaces.reserve(static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors));

    uint32_t totalNeighbors = 0;

    for (uint32_t cellIdx = 0; cellIdx < resources.voronoiNodeCount; ++cellIdx) {
        uint32_t neighborOffset = totalNeighbors;
        uint32_t validNeighborCount = 0;
        if ((seedFlags[cellIdx] & 1u) != 0u) {
            nodes[cellIdx].neighborOffset = neighborOffset;
            nodes[cellIdx].neighborCount = 0;
            continue;
        }

        uint32_t interfaceCount = nodes[cellIdx].interfaceNeighborCount;
        if (interfaceCount > maxNeighbors) {
            interfaceCount = maxNeighbors;
        }

        const glm::vec3 cellPosition(seedPositions[cellIdx]);

        for (uint32_t k = 0; k < interfaceCount; ++k) {
            uint32_t idx = cellIdx * maxNeighbors + k;
            uint32_t neighborIdx = neighborIds[idx];
            float area = areas[idx];
            if (neighborIdx == UINT32_MAX || area <= 0.0f) {
                continue;
            }

            const glm::vec3 seedB(seedPositions[neighborIdx]);
            const float distance = glm::length(seedB - cellPosition);
            if (distance <= 1e-12f || area <= 1e-8f) {
                continue;
            }

            const float conductance = area / distance;

            voronoi::GMLSInterface interfaceSample{};
            interfaceSample.neighborIdx = neighborIdx;
            interfaceSample.conductance = conductance;

            interfaces.push_back(interfaceSample);
            ++validNeighborCount;
            ++totalNeighbors;
        }

        nodes[cellIdx].neighborOffset = neighborOffset;
        nodes[cellIdx].neighborCount = validNeighborCount;
    }



    if (interfaces.empty()) {
        memoryAllocator.free(areasStagingBuf, areasStagingOff);
        memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
        memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
        memoryAllocator.free(seedPosStagingBuf, seedPosStagingOff);
        return false;
    }

    {
        VkDeviceSize bufferSize = sizeof(voronoi::GMLSInterface) * interfaces.size();
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) {
            memoryAllocator.free(areasStagingBuf, areasStagingOff);
            memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
            memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
            memoryAllocator.free(seedPosStagingBuf, seedPosStagingOff);
            return false;
        }
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        std::memcpy(mapped, interfaces.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) {
            memoryAllocator.free(stagingBuf, stagingOff);
            memoryAllocator.free(areasStagingBuf, areasStagingOff);
            memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);
            memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
            memoryAllocator.free(seedPosStagingBuf, seedPosStagingOff);
            return false;
        }
        resources.gmlsInterfaceBuffer = buf;
        resources.gmlsInterfaceBufferOffset = off;

        VkCommandBuffer cmd2 = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd2, stagingBuf, resources.gmlsInterfaceBuffer, 1, &region);
        renderCommandPool.endCommands(cmd2);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    memoryAllocator.free(areasStagingBuf, areasStagingOff);
    memoryAllocator.free(neighborIdsStagingBuf, neighborIdsStagingOff);

    // Write back modified nodes 
    {
        VkCommandBuffer cmd3 = renderCommandPool.beginCommands();
        VkBufferCopy nodeWriteRegion{nodeStagingOff, resources.voronoiNodeBufferOffset, nodeBufferSize};
        vkCmdCopyBuffer(cmd3, nodeStagingBuf, resources.voronoiNodeBuffer, 1, &nodeWriteRegion);
        renderCommandPool.endCommands(cmd3);
    }

    memoryAllocator.free(nodeStagingBuf, nodeStagingOff);
    memoryAllocator.free(seedPosStagingBuf, seedPosStagingOff);
    return true;
}

bool VoronoiSystemBuildStage::rebuildOccupancyPointBuffer(VoronoiSystemRuntime& runtime) const {
    if (resources.occupancyPointBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(resources.occupancyPointBuffer, resources.occupancyPointBufferOffset);
        resources.occupancyPointBuffer = VK_NULL_HANDLE;
        resources.occupancyPointBufferOffset = 0;
    }
    resources.occupancyPointCount = 0;

    if (!runtime.isVoxelGridBuilt()) {
        return true;
    }

    const VoxelGrid& voxelGrid = runtime.getVoxelGrid();
    const auto& occupancy = voxelGrid.getOccupancyData();
    const auto& params = voxelGrid.getParams();
    VoronoiModelRuntime* modelRuntime = runtime.getModelRuntime();
    const glm::mat4 modelMatrix = modelRuntime ? modelRuntime->getModelMatrix() : glm::mat4(1.0f);

    size_t estimatedPointCount = occupancy.size() / 4;
    std::vector<PointRenderer::PointVertex> points;
    points.reserve(estimatedPointCount);

    const int dimX = params.gridDim.x;
    const int dimY = params.gridDim.y;
    const int dimZ = params.gridDim.z;
    const int stride = dimX + 1;

    for (int z = 0; z <= dimZ; ++z) {
        for (int y = 0; y <= dimY; ++y) {
            for (int x = 0; x <= dimX; ++x) {
                const size_t idx =
                    static_cast<size_t>(z) * static_cast<size_t>(stride) * static_cast<size_t>(stride) +
                    static_cast<size_t>(y) * static_cast<size_t>(stride) +
                    static_cast<size_t>(x);
                if (idx >= occupancy.size()) {
                    continue;
                }

                const uint8_t occ = occupancy[idx];
                if (occ == 0) {
                    continue;
                }

                const glm::vec3 localPos = voxelGrid.getCornerPosition(x, y, z);
                const glm::vec3 worldPos = glm::vec3(modelMatrix * glm::vec4(localPos, 1.0f));
                glm::vec3 color = glm::vec3(0.2f, 1.0f, 0.2f);
                if (occ == 1) {
                    color = glm::vec3(1.0f, 0.2f, 0.2f);
                }
                points.push_back({ worldPos, color });
            }
        }
    }

    if (points.empty()) {
        return true;
    }

    const VkDeviceSize bufferSize = sizeof(PointRenderer::PointVertex) * points.size();
    auto [buffer, offset] = memoryAllocator.allocate(
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        alignof(PointRenderer::PointVertex));
    if (buffer == VK_NULL_HANDLE) {
        return false;
    }

    void* mappedPtr = memoryAllocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        memoryAllocator.free(buffer, offset);
        return false;
    }

    std::memcpy(mappedPtr, points.data(), static_cast<size_t>(bufferSize));
    resources.occupancyPointBuffer = buffer;
    resources.occupancyPointBufferOffset = offset;
    resources.occupancyPointCount = static_cast<uint32_t>(points.size());
    return true;
}

bool VoronoiSystemBuildStage::dispatchVoronoiCompute(VoronoiSystemRuntime& runtime, bool debugEnable, uint32_t maxNeighbors) {
    VoronoiModelRuntime* modelRuntime = runtime.getModelRuntime();
    if (!modelRuntime || !runtime.getIntegrator()) {
        return false;
    }

    const auto& domainSeeds = runtime.getSeedPositions();
    if (domainSeeds.empty()) {
        return false;
    }
    if (runtime.getSeedFlags().size() != domainSeeds.size()) {
        return false;
    }

    resources.voronoiNodeCount = static_cast<uint32_t>(domainSeeds.size());
    if (resources.voronoiNodeCount == 0) {
        return false;
    }

    std::vector<voronoi::Node> initialNodes(resources.voronoiNodeCount);
    for (voronoi::Node& node : initialNodes) {
        node.volume = 0.0f;
        node.neighborOffset = 0u;
        node.neighborCount = 0u;
        node.interfaceNeighborCount = 0u;
    }

    std::vector<glm::vec4> globalSeedPositions;
    std::vector<uint32_t> globalSeedFlags;
    std::vector<uint32_t> globalNeighborIndices;
    globalSeedPositions.reserve(resources.voronoiNodeCount);
    globalSeedFlags.reserve(resources.voronoiNodeCount);
    globalNeighborIndices.reserve(
        static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors));

    const auto& domainNeighbors = runtime.getNeighborIndices();
    const size_t expectedNeighborCount =
        static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors);
    if (domainNeighbors.size() < expectedNeighborCount) {
        return false;
    }

    globalSeedPositions.insert(globalSeedPositions.end(), domainSeeds.begin(), domainSeeds.end());
    globalSeedFlags.insert(globalSeedFlags.end(), runtime.getSeedFlags().begin(), runtime.getSeedFlags().end());

    for (uint32_t localNodeIndex = 0; localNodeIndex < resources.voronoiNodeCount; ++localNodeIndex) {
        const size_t base = static_cast<size_t>(localNodeIndex) * static_cast<size_t>(maxNeighbors);
        for (uint32_t k = 0; k < maxNeighbors; ++k) {
            uint32_t neighborIndex = domainNeighbors[base + static_cast<size_t>(k)];
            if (neighborIndex != UINT32_MAX) {
                if (neighborIndex >= resources.voronoiNodeCount) {
                    neighborIndex = UINT32_MAX;
                }
            }
            globalNeighborIndices.push_back(neighborIndex);
        }
    }

    if (globalSeedPositions.size() != resources.voronoiNodeCount ||
        globalSeedFlags.size() != resources.voronoiNodeCount) {
        return false;
    }

    if (!createGeometryBuffers(
            initialNodes,
            globalSeedPositions,
            globalSeedFlags,
            globalNeighborIndices,
            debugEnable,
            maxNeighbors)) {
        return false;
    }

    if (voronoiGeoCompute) {
        voronoiGeoCompute->initialize(resources.voronoiNodeCount);
    }

    freeBuffer(resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);

    const auto& meshTris = runtime.getMeshTriangles();
    if (meshTris.empty()) {
        return false;
    }

    {
        VkDeviceSize bufferSize = sizeof(glm::vec4) * 3 * meshTris.size();
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, meshTris.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.meshTriangleBuffer = buf;
        resources.meshTriangleBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.meshTriangleBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    VoxelGrid::VoxelGridParams params{};
    std::vector<uint32_t> occupancy32;
    std::vector<int32_t> trianglesList;
    std::vector<int32_t> offsets;

    if (runtime.isVoxelGridBuilt()) {
        params = runtime.getVoxelGrid().getParams();
        const auto& occupancy8 = runtime.getVoxelGrid().getOccupancyData();
        occupancy32.resize(occupancy8.size());
        for (size_t i = 0; i < occupancy8.size(); ++i) {
            occupancy32[i] = static_cast<uint32_t>(occupancy8[i]);
        }
        trianglesList = runtime.getVoxelGrid().getTrianglesList();
        offsets = runtime.getVoxelGrid().getOffsets();
    } else {
        params.gridMin = glm::vec3(0.0f);
        params.cellSize = 1.0f;
        params.gridDim = glm::ivec3(1);
        params.totalCells = 1;
    }

    if (occupancy32.empty()) {
        occupancy32.push_back(0u);
    }
    if (trianglesList.empty()) {
        trianglesList.push_back(-1);
    }
    if (offsets.empty()) {
        offsets.push_back(0);
    }

    {
        VkDeviceSize bufferSize = sizeof(VoxelGrid::VoxelGridParams);
        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (buf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(buf, off);
        if (!mapped) { memoryAllocator.free(buf, off); return false; }
        std::memcpy(mapped, &params, sizeof(VoxelGrid::VoxelGridParams));
        resources.voxelGridParamsBuffer = buf;
        resources.voxelGridParamsBufferOffset = off;
    }

    {
        VkDeviceSize bufferSize = sizeof(uint32_t) * occupancy32.size();
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, occupancy32.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.voxelOccupancyBuffer = buf;
        resources.voxelOccupancyBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.voxelOccupancyBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    {
        VkDeviceSize bufferSize = sizeof(int32_t) * trianglesList.size();
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, trianglesList.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.voxelTrianglesListBuffer = buf;
        resources.voxelTrianglesListBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.voxelTrianglesListBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    {
        VkDeviceSize bufferSize = sizeof(int32_t) * offsets.size();
        auto [stagingBuf, stagingOff] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingBuf == VK_NULL_HANDLE) return false;
        void* mapped = memoryAllocator.getMappedPointer(stagingBuf, stagingOff);
        if (!mapped) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        std::memcpy(mapped, offsets.data(), static_cast<size_t>(bufferSize));

        auto [buf, off] = memoryAllocator.allocate(
            bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (buf == VK_NULL_HANDLE) { memoryAllocator.free(stagingBuf, stagingOff); return false; }
        resources.voxelOffsetsBuffer = buf;
        resources.voxelOffsetsBufferOffset = off;

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{stagingOff, off, bufferSize};
        vkCmdCopyBuffer(cmd, stagingBuf, resources.voxelOffsetsBuffer, 1, &region);
        renderCommandPool.endCommands(cmd);
        memoryAllocator.free(stagingBuf, stagingOff);
    }

    if (voronoiGeoCompute) {
        VoronoiGeoCompute::Bindings bindings{};
        bindings.voronoiNodeBuffer = resources.voronoiNodeBuffer;
        bindings.voronoiNodeBufferOffset = resources.voronoiNodeBufferOffset;
        bindings.voronoiNodeBufferRange = sizeof(voronoi::Node) * resources.voronoiNodeCount;
        bindings.meshTriangleBuffer = resources.meshTriangleBuffer;
        bindings.meshTriangleBufferOffset = resources.meshTriangleBufferOffset;
        bindings.seedPositionBuffer = resources.seedPositionBuffer;
        bindings.seedPositionBufferOffset = resources.seedPositionBufferOffset;
        bindings.voxelGridParamsBuffer = resources.voxelGridParamsBuffer;
        bindings.voxelGridParamsBufferOffset = resources.voxelGridParamsBufferOffset;
        bindings.voxelGridParamsBufferRange = sizeof(VoxelGrid::VoxelGridParams);
        bindings.voxelOccupancyBuffer = resources.voxelOccupancyBuffer;
        bindings.voxelOccupancyBufferOffset = resources.voxelOccupancyBufferOffset;
        bindings.voxelTrianglesListBuffer = resources.voxelTrianglesListBuffer;
        bindings.voxelTrianglesListBufferOffset = resources.voxelTrianglesListBufferOffset;
        bindings.voxelOffsetsBuffer = resources.voxelOffsetsBuffer;
        bindings.voxelOffsetsBufferOffset = resources.voxelOffsetsBufferOffset;
        bindings.neighborIndicesBuffer = resources.neighborIndicesBuffer;
        bindings.neighborIndicesBufferOffset = resources.neighborIndicesBufferOffset;
        bindings.interfaceAreasBuffer = resources.interfaceAreasBuffer;
        bindings.interfaceAreasBufferOffset = resources.interfaceAreasBufferOffset;
        bindings.interfaceNeighborIdsBuffer = resources.interfaceNeighborIdsBuffer;
        bindings.interfaceNeighborIdsBufferOffset = resources.interfaceNeighborIdsBufferOffset;
        bindings.debugCellGeometryBuffer = resources.debugCellGeometryBuffer;
        bindings.debugCellGeometryBufferOffset = resources.debugCellGeometryBufferOffset;
        bindings.seedFlagsBuffer = resources.seedFlagsBuffer;
        bindings.seedFlagsBufferOffset = resources.seedFlagsBufferOffset;
        bindings.voronoiDumpBuffer = resources.voronoiDumpBuffer;
        bindings.voronoiDumpBufferOffset = resources.voronoiDumpBufferOffset;

        voronoiGeoCompute->updateDescriptors(bindings);

        VoronoiGeoCompute::PushConstants geoPushConstants{};
        geoPushConstants.debugEnable = debugEnable ? 1u : 0u;
        geoPushConstants.nodeOffset = 0;
        geoPushConstants.nodeCount = resources.voronoiNodeCount;
        voronoiGeoCompute->dispatch(geoPushConstants);
    }

    freeBuffer(resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);

    setGhostFromVolumes(runtime);

    if (VoronoiModelRuntime* modelRuntime = runtime.getModelRuntime()) {
        modelRuntime->setStencilKDTree(nullptr);
    }

    if (!buildGMLSInterfaceBuffer(runtime, maxNeighbors)) {
        return false;
    }

    return rebuildOccupancyPointBuffer(runtime);
}

bool VoronoiSystemBuildStage::stageSurfaceMappings(VoronoiSystemRuntime& runtime) const {
    VoronoiModelRuntime* modelRuntime = runtime.getModelRuntime();
    if (!modelRuntime || !runtime.getIntegrator()) {
        return false;
    }


    const auto& domainSeedPositions4 = runtime.getSeedPositions();
    if (domainSeedPositions4.empty() || runtime.getSeedFlags().size() != domainSeedPositions4.size()) {
        return false;
    }

    StencilKDTree* kdTreePtr = modelRuntime->getStencilKDTree();
    if (!kdTreePtr) {
        modelRuntime->setStencilKDTree(std::make_unique<StencilKDTree>(runtime.getSeedFlags(), domainSeedPositions4));
        kdTreePtr = modelRuntime->getStencilKDTree();
    }
    
    if (!kdTreePtr || !kdTreePtr->isValid()) {
        return false;
    }
    const StencilKDTree& kdTree = *kdTreePtr;

    const size_t targetSupportCount = 32;
    const size_t supportCount = kdTree.supportCount;

    std::vector<glm::vec3> surfacePoints = modelRuntime->getIntrinsicSurfacePositions();
    if (surfacePoints.empty()) {
        return false;
    }

    std::vector<voronoi::GMLSSurfaceStencil> stencils(surfacePoints.size());
    std::vector<voronoi::GMLSSurfaceWeight> valueWeights;
    std::vector<voronoi::GMLSSurfaceGradientWeight> gradientWeights;
    valueWeights.reserve(surfacePoints.size() * targetSupportCount);
    gradientWeights.reserve(surfacePoints.size() * targetSupportCount);

    std::vector<size_t> retIndices(supportCount, 0);
    std::vector<float> outDistSq(supportCount, 0.0f);
    std::vector<glm::dvec3> sourcePositions;
    std::vector<uint32_t> sourceGlobalIndices;
    std::vector<double> valueWeightDoubles;
    std::vector<glm::dvec3> gradientWeightTriples;
    sourcePositions.reserve(supportCount);
    sourceGlobalIndices.reserve(supportCount);

    for (size_t vertexIndex = 0; vertexIndex < surfacePoints.size(); ++vertexIndex) {
        const glm::vec3& point = surfacePoints[vertexIndex];
        const float query[3] = { point.x, point.y, point.z };

        nanoflann::KNNResultSet<float> resultSet(supportCount);
        resultSet.init(retIndices.data(), outDistSq.data());
        kdTree.index.findNeighbors(resultSet, query);

        sourcePositions.clear();
        sourceGlobalIndices.clear();
        float maxDistSq = 0.0f;
        for (size_t neighborIndex = 0; neighborIndex < supportCount; ++neighborIndex) {
            const size_t localIdx = retIndices[neighborIndex];
            const glm::vec3& seedPos = kdTree.regularSeedPositions[localIdx];

            if (runtime.isVoxelGridBuilt()) {
                if (!runtime.getVoxelGrid().segmentStaysInside(point, seedPos, 8)) {
                    continue;
                }
            }

            sourcePositions.push_back(glm::dvec3(seedPos));
            sourceGlobalIndices.push_back(kdTree.regularLocalIndices[localIdx]);
            if (outDistSq[neighborIndex] > maxDistSq) maxDistSq = outDistSq[neighborIndex];

            if (sourcePositions.size() >= targetSupportCount) {
                break;
            }
        }

        if (sourcePositions.size() < 4) {
            continue;
        }

        const double kernelRadius = std::max<double>(static_cast<double>(std::sqrt(maxDistSq)) * 2.0, 1e-5);
        const bool validWeights = GMLS::computeWeights(glm::dvec3(point), sourcePositions, kernelRadius, valueWeightDoubles, gradientWeightTriples);

        voronoi::GMLSSurfaceStencil& stencil = stencils[vertexIndex];
        stencil.valueWeightOffset = static_cast<uint32_t>(valueWeights.size());
        stencil.gradientWeightOffset = static_cast<uint32_t>(gradientWeights.size());
        stencil.valueWeightCount = 0;
        stencil.gradientWeightCount = 0;

        if (!validWeights) {
            continue;
        }

        for (size_t neighborIndex = 0; neighborIndex < sourcePositions.size(); ++neighborIndex) {
            const uint32_t globalCellIndex = sourceGlobalIndices[neighborIndex];
            const float valueWeight = static_cast<float>(valueWeightDoubles[neighborIndex]);
            const glm::dvec3 gradientWeight = gradientWeightTriples[neighborIndex];

            if (std::abs(valueWeight) > 1e-7f) {
                valueWeights.push_back({ globalCellIndex, valueWeight, 0u, 0u });
                ++stencil.valueWeightCount;
            }

            if (glm::dot(gradientWeight, gradientWeight) > 1e-14) {
                gradientWeights.push_back({
                    globalCellIndex,
                    static_cast<float>(gradientWeight.x),
                    static_cast<float>(gradientWeight.y),
                    static_cast<float>(gradientWeight.z)
                });
                ++stencil.gradientWeightCount;
            }
        }
    }

    modelRuntime->stageGMLSSurfaceData(stencils, valueWeights, gradientWeights);
    return true;
}

void VoronoiSystemBuildStage::freeBuffer(VkBuffer& buffer, VkDeviceSize& offset) {
    if (buffer != VK_NULL_HANDLE) {
        memoryAllocator.free(buffer, offset);
        buffer = VK_NULL_HANDLE;
        offset = 0;
    }
}

void VoronoiSystemBuildStage::cleanupResources() {
    if (voronoiGeoCompute) {
        voronoiGeoCompute->cleanupResources();
    }
}
