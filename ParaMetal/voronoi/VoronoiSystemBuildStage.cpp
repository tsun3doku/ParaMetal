#include "VoronoiSystemBuildStage.hpp"

#include "spatial/VoxelGrid.hpp"
#include "voronoi/VoronoiDomainRuntime.hpp"
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
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resources(resources),
      commandPool(commandPool) {
    initializeVoronoiGeoCompute();
}

VoronoiSystemBuildStage::~VoronoiSystemBuildStage() {
}

void VoronoiSystemBuildStage::initializeVoronoiGeoCompute() {
    voronoiGeoCompute = std::make_unique<VoronoiGeoCompute>(vulkanDevice, commandPool);
}

bool VoronoiSystemBuildStage::buildVoronoiDiagram(
    VoronoiSystemRuntime& runtime,
    float cellSize,
    int voxelResolution,
    uint32_t maxNeighbors) const {
    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime) {
        return false;
    }

    const auto& seedPositions = runtime.getSeedPositions();
    if (seedPositions.empty()) {
        return false;
    }

    runtime.resetIntegrator();
    if (!runtime.getIntegrator()) {
        return false;
    }

    if (domainRuntime->isPointDomain()) {
        std::vector<glm::dvec3> dSeedPositions;
        dSeedPositions.reserve(seedPositions.size());
        runtime.getSeedFlags().clear();
        runtime.getSeedFlags().reserve(seedPositions.size());
        for (const auto& pos : seedPositions) {
            dSeedPositions.push_back(glm::dvec3(pos));
            runtime.getSeedFlags().push_back(0u);
        }

        runtime.getIntegrator()->computeNeighbors(dSeedPositions, maxNeighbors, runtime.getSeedPositions(), runtime.getNeighborIndices());
        runtime.getMeshTriangles().clear();
        return true;
    }

    auto* modelRuntime = static_cast<VoronoiModelRuntime*>(domainRuntime);
    const uint32_t receiverModelId = modelRuntime->getRuntimeModelId();
    if (receiverModelId == 0) {
        return false;
    }

    runtime.resetMeshGrid();
    if (!runtime.getMeshGrid()) {
        return false;
    }

    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh = modelRuntime->getIntrinsicMesh();
    const std::vector<glm::vec3>& geometryPositions = modelRuntime->getGeometryPositions();
    const std::vector<uint32_t>& geometryIndices = modelRuntime->getGeometryTriangleIndices();
    runtime.getMeshGrid()->buildGrids(
        intrinsicMesh,
        geometryPositions,
        geometryIndices,
        cellSize,
        runtime.getVoxelGrid(),
        voxelResolution);
    runtime.setVoxelGridBuilt(true);

    std::vector<glm::dvec3> dSeedPositions;
    dSeedPositions.reserve(seedPositions.size());
    runtime.getSeedFlags().clear();
    runtime.getSeedFlags().reserve(seedPositions.size());
    for (const auto& pos : seedPositions) {
        dSeedPositions.push_back(glm::dvec3(pos));
        runtime.getSeedFlags().push_back(0u);
    }

    runtime.getIntegrator()->computeNeighbors(dSeedPositions, maxNeighbors, runtime.getSeedPositions(), runtime.getNeighborIndices());
    runtime.getIntegrator()->extractMeshTriangles(geometryPositions, geometryIndices, runtime.getMeshTriangles());

    return true;
}

void VoronoiSystemBuildStage::setGhostFromVolumes(VoronoiSystemRuntime& runtime) {
    if (resources.voronoiNodeCount == 0 || !resources.voronoiSeedFlagsBuffer) {
        return;
    }

    std::vector<voronoi::Node> nodes(resources.voronoiNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool,
        resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset,
        nodes.size() * sizeof(voronoi::Node), nodes.data()) != VK_SUCCESS) {
        return;
    }

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
}

void VoronoiSystemBuildStage::setGhostFromVoxelGrid(VoronoiSystemRuntime& runtime) {
    if (!runtime.isVoxelGridBuilt() || !runtime.getMeshGrid()) {
        return;
    }
    const auto& seedPositions = runtime.getSeedPositions();
    if (seedPositions.size() != runtime.getSeedFlags().size()) {
        return;
    }
    const float maxDistFromSurface = runtime.getMeshGrid()->getCellSize() * 1.5f;
    uint32_t ghostCount = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(seedPositions.size()); ++i) {
        const glm::vec4& seed = seedPositions[i];
        glm::ivec3 voxel = runtime.getVoxelGrid().worldToVoxel(glm::vec3(seed));
        uint8_t occ = runtime.getVoxelGrid().getOccupancy(voxel.x, voxel.y, voxel.z);
        bool isInside = (occ == 2 || occ == 1);
        float distToSurface = runtime.getMeshGrid()->sampleSDFGrid(glm::vec3(seed));
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

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            initialNodes.data(),
            sizeof(voronoi::Node) * resources.voronoiNodeCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.voronoiNodeBuffer,
            resources.voronoiNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t> flattenedNeighbors = neighborIndices;
    if (flattenedNeighbors.empty()) {
        flattenedNeighbors.resize(
            static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors),
            UINT32_MAX);
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            flattenedNeighbors.data(),
            sizeof(uint32_t) * flattenedNeighbors.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.voronoiNeighborIndicesBuffer,
            resources.voronoiNeighborIndicesBufferOffset) != VK_SUCCESS) {
        return false;
    }

    const size_t interfaceDataSize = static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors);

    std::vector<float> emptyAreas(interfaceDataSize, 0.0f);
    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            emptyAreas.data(),
            sizeof(float) * interfaceDataSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.voronoiInterfaceAreasBuffer,
            resources.voronoiInterfaceAreasBufferOffset) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t> emptyIds(interfaceDataSize, UINT32_MAX);
    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            emptyIds.data(),
            sizeof(uint32_t) * interfaceDataSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.voronoiInterfaceNeighborIdsBuffer,
            resources.voronoiInterfaceNeighborIdsBufferOffset) != VK_SUCCESS) {
        return false;
    }

    uint32_t numDebugCells = debugEnable ? resources.voronoiNodeCount : 1u;
    std::vector<voronoi::DebugCellGeometry> debugCells(numDebugCells);
    for (auto& cell : debugCells) {
        cell.cellID = 0;
        cell.vertexCount = 0;
        cell.triangleCount = 0;
        cell.volume = 0.0f;
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            debugCells.data(),
            sizeof(voronoi::DebugCellGeometry) * numDebugCells,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.debugCellGeometryBuffer,
            resources.debugCellGeometryBufferOffset) != VK_SUCCESS) {
        return false;
    }

    uint32_t dumpCount = debugEnable ? voronoi::DEBUG_DUMP_CELL_COUNT : 1u;
    std::vector<voronoi::DumpInfo> dumpInfos(dumpCount);
    std::memset(dumpInfos.data(), 0, sizeof(voronoi::DumpInfo) * dumpCount);

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            dumpInfos.data(),
            sizeof(voronoi::DumpInfo) * dumpCount,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.voronoiDumpBuffer,
            resources.voronoiDumpBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            seedPositions.data(),
            sizeof(glm::vec4) * seedPositions.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            resources.seedPositionBuffer,
            resources.seedPositionBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            seedFlags.data(),
            sizeof(uint32_t) * seedFlags.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.voronoiSeedFlagsBuffer,
            resources.voronoiSeedFlagsBufferOffset) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VoronoiSystemBuildStage::buildGMLSInterfaceBuffer(VoronoiSystemRuntime& runtime, uint32_t maxNeighbors) {
    if (resources.voronoiNodeCount == 0) {
        return false;
    }

    std::vector<float> areas(static_cast<size_t>(resources.voronoiNodeCount) * maxNeighbors);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, resources.voronoiInterfaceAreasBuffer, resources.voronoiInterfaceAreasBufferOffset, 
        areas.size() * sizeof(float), areas.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t> neighborIds(static_cast<size_t>(resources.voronoiNodeCount) * maxNeighbors);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, resources.voronoiInterfaceNeighborIdsBuffer, resources.voronoiInterfaceNeighborIdsBufferOffset, 
        neighborIds.size() * sizeof(uint32_t), neighborIds.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<voronoi::Node> nodes(resources.voronoiNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset, 
        nodes.size() * sizeof(voronoi::Node), nodes.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<glm::vec4> seedPositions(resources.voronoiNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, resources.seedPositionBuffer, resources.seedPositionBufferOffset, 
        seedPositions.size() * sizeof(glm::vec4), seedPositions.data()) != VK_SUCCESS) {
        return false;
    }

    const std::vector<uint32_t>& seedFlags = runtime.getSeedFlags();

    std::vector<voronoi::GMLSInterface> interfaces;
    interfaces.reserve(static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors));
    std::vector<voronoi::GMLSInterface> cellInterfaces;
    cellInterfaces.reserve(maxNeighbors);

    uint32_t totalNeighbors = 0;
    for (uint32_t cellIdx = 0; cellIdx < resources.voronoiNodeCount; ++cellIdx) {
        uint32_t neighborOffset = totalNeighbors;
        cellInterfaces.clear();
        
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
            const uint32_t idx = cellIdx * maxNeighbors + k;
            const uint32_t neighborIdx = neighborIds[idx];
            const float area = areas[idx];
            
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

            cellInterfaces.push_back(interfaceSample);
        }

        std::sort(
            cellInterfaces.begin(),
            cellInterfaces.end(),
            [](const voronoi::GMLSInterface& a, const voronoi::GMLSInterface& b) {
                return a.neighborIdx < b.neighborIdx;
            });

        interfaces.insert(interfaces.end(), cellInterfaces.begin(), cellInterfaces.end());
        nodes[cellIdx].neighborOffset = neighborOffset;
        nodes[cellIdx].neighborCount = static_cast<uint32_t>(cellInterfaces.size());
        totalNeighbors += static_cast<uint32_t>(cellInterfaces.size());
    }

    if (interfaces.empty()) {
        uint32_t ghostedCount = 0;
        uint32_t zeroInterfaceCount = 0;
        for (uint32_t i = 0; i < resources.voronoiNodeCount; ++i) {
            if ((seedFlags[i] & 1u) != 0u) {
                ++ghostedCount;
            } else if (nodes[i].interfaceNeighborCount == 0) {
                ++zeroInterfaceCount;
            }
        }
        return false;
    }

    resources.voronoiGMLSInterfaceBuffer = VK_NULL_HANDLE;
    resources.voronoiGMLSInterfaceBufferOffset = 0;
    VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    
    if (uploadDeviceBuffer(memoryAllocator, commandPool, interfaces.data(), 
        sizeof(voronoi::GMLSInterface) * interfaces.size(), 
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment, 
        resources.voronoiGMLSInterfaceBuffer, resources.voronoiGMLSInterfaceBufferOffset) != VK_SUCCESS) {
        return false;
    }

    resources.voronoiNodeBuffer = VK_NULL_HANDLE;
    resources.voronoiNodeBufferOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, nodes.data(),
        sizeof(voronoi::Node) * nodes.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
        resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }

    resources.voronoiInterfaceAreasBuffer = VK_NULL_HANDLE;
    resources.voronoiInterfaceAreasBufferOffset = 0;
    resources.voronoiInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
    resources.voronoiInterfaceNeighborIdsBufferOffset = 0;

    return true;
}

bool VoronoiSystemBuildStage::rebuildOccupancyPointBuffer(VoronoiSystemRuntime& runtime) const {
    if (resources.occupancyPointBuffer != VK_NULL_HANDLE) {
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
    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    auto* modelRuntime = domainRuntime ? static_cast<VoronoiModelRuntime*>(domainRuntime) : nullptr;
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
    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime || !runtime.getIntegrator()) {
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

    if (domainRuntime->isPointDomain()) {
        // Point domain: no geo compute. Build interfaces from distances directly.
        std::vector<voronoi::GMLSInterface> interfaces;
        interfaces.reserve(static_cast<size_t>(resources.voronoiNodeCount) * static_cast<size_t>(maxNeighbors));
        std::vector<voronoi::GMLSInterface> cellInterfaces;
        cellInterfaces.reserve(maxNeighbors);

        uint32_t totalNeighbors = 0;
        for (uint32_t cellIdx = 0; cellIdx < resources.voronoiNodeCount; ++cellIdx) {
            uint32_t neighborOffset = totalNeighbors;
            cellInterfaces.clear();

            if ((globalSeedFlags[cellIdx] & 1u) != 0u) {
                initialNodes[cellIdx].neighborOffset = neighborOffset;
                initialNodes[cellIdx].neighborCount = 0;
                continue;
            }

            const glm::vec3 cellPosition(globalSeedPositions[cellIdx]);

            for (uint32_t k = 0; k < maxNeighbors; ++k) {
                const uint32_t idx = cellIdx * maxNeighbors + k;
                uint32_t neighborIdx = globalNeighborIndices[idx];
                if (neighborIdx == UINT32_MAX || neighborIdx >= resources.voronoiNodeCount) {
                    continue;
                }

                const glm::vec3 seedB(globalSeedPositions[neighborIdx]);
                const float distance = glm::length(seedB - cellPosition);
                if (distance <= 1e-12f) {
                    continue;
                }

                const float conductance = 1.0f / distance;
                cellInterfaces.push_back({neighborIdx, conductance});
            }

            std::sort(
                cellInterfaces.begin(),
                cellInterfaces.end(),
                [](const voronoi::GMLSInterface& a, const voronoi::GMLSInterface& b) {
                    return a.neighborIdx < b.neighborIdx;
                });

            interfaces.insert(interfaces.end(), cellInterfaces.begin(), cellInterfaces.end());
            initialNodes[cellIdx].neighborOffset = neighborOffset;
            initialNodes[cellIdx].neighborCount = static_cast<uint32_t>(cellInterfaces.size());
            initialNodes[cellIdx].interfaceNeighborCount = initialNodes[cellIdx].neighborCount;
            totalNeighbors += static_cast<uint32_t>(cellInterfaces.size());
        }

        if (interfaces.empty()) {
            return false;
        }

        resources.voronoiGMLSInterfaceBuffer = VK_NULL_HANDLE;
        resources.voronoiGMLSInterfaceBufferOffset = 0;
        VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

        if (uploadDeviceBuffer(memoryAllocator, commandPool, interfaces.data(),
            sizeof(voronoi::GMLSInterface) * interfaces.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
            resources.voronoiGMLSInterfaceBuffer, resources.voronoiGMLSInterfaceBufferOffset) != VK_SUCCESS) {
            return false;
        }

        resources.voronoiNodeBuffer = VK_NULL_HANDLE;
        resources.voronoiNodeBufferOffset = 0;
        if (uploadDeviceBuffer(memoryAllocator, commandPool, initialNodes.data(),
            sizeof(voronoi::Node) * initialNodes.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
            resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset) != VK_SUCCESS) {
            return false;
        }

        resources.voronoiInterfaceAreasBuffer = VK_NULL_HANDLE;
        resources.voronoiInterfaceAreasBufferOffset = 0;
        resources.voronoiInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
        resources.voronoiInterfaceNeighborIdsBufferOffset = 0;

        runtime.buildSimSpaceMapping();
        if (!runtime.buildSimBuffers(memoryAllocator, commandPool)) {
            return false;
        }

        return true;
    }

    // Mesh domain path
    auto* modelRuntime = static_cast<VoronoiModelRuntime*>(domainRuntime);
    (void)modelRuntime;

    if (voronoiGeoCompute) {
        voronoiGeoCompute->initialize(resources.voronoiNodeCount);
    }

    freeBuffer(memoryAllocator, resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);

    const auto& meshTris = runtime.getMeshTriangles();
    if (meshTris.empty()) {
        return false;
    }

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            meshTris.data(),
            sizeof(glm::vec4) * 3 * meshTris.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.meshTriangleBuffer,
            resources.meshTriangleBufferOffset) != VK_SUCCESS) {
        return false;
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
        void* mapped = nullptr;
        if (createUniformBuffer(
                memoryAllocator,
                vulkanDevice,
                sizeof(VoxelGrid::VoxelGridParams),
                resources.voxelGridParamsBuffer,
                resources.voxelGridParamsBufferOffset,
                &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(mapped, &params, sizeof(VoxelGrid::VoxelGridParams));
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            occupancy32.data(),
            sizeof(uint32_t) * occupancy32.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.voxelOccupancyBuffer,
            resources.voxelOccupancyBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            trianglesList.data(),
            sizeof(int32_t) * trianglesList.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.voxelTrianglesListBuffer,
            resources.voxelTrianglesListBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            offsets.data(),
            sizeof(int32_t) * offsets.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            resources.voxelOffsetsBuffer,
            resources.voxelOffsetsBufferOffset) != VK_SUCCESS) {
        return false;
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
        bindings.voronoiNeighborIndicesBuffer = resources.voronoiNeighborIndicesBuffer;
        bindings.voronoiNeighborIndicesBufferOffset = resources.voronoiNeighborIndicesBufferOffset;
        bindings.voronoiInterfaceAreasBuffer = resources.voronoiInterfaceAreasBuffer;
        bindings.voronoiInterfaceAreasBufferOffset = resources.voronoiInterfaceAreasBufferOffset;
        bindings.voronoiInterfaceNeighborIdsBuffer = resources.voronoiInterfaceNeighborIdsBuffer;
        bindings.voronoiInterfaceNeighborIdsBufferOffset = resources.voronoiInterfaceNeighborIdsBufferOffset;
        bindings.debugCellGeometryBuffer = resources.debugCellGeometryBuffer;
        bindings.debugCellGeometryBufferOffset = resources.debugCellGeometryBufferOffset;
        bindings.voronoiSeedFlagsBuffer = resources.voronoiSeedFlagsBuffer;
        bindings.voronoiSeedFlagsBufferOffset = resources.voronoiSeedFlagsBufferOffset;
        bindings.voronoiDumpBuffer = resources.voronoiDumpBuffer;
        bindings.voronoiDumpBufferOffset = resources.voronoiDumpBufferOffset;

        voronoiGeoCompute->updateDescriptors(bindings);

        VoronoiGeoCompute::PushConstants geoPushConstants{};
        geoPushConstants.debugEnable = debugEnable ? 1u : 0u;
        geoPushConstants.nodeOffset = 0;
        geoPushConstants.nodeCount = resources.voronoiNodeCount;
        voronoiGeoCompute->dispatch(geoPushConstants);
    }

    freeBuffer(memoryAllocator, resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(memoryAllocator, resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);

    setGhostFromVolumes(runtime);
    modelRuntime->setStencilKDTree(nullptr);

    if (!buildGMLSInterfaceBuffer(runtime, maxNeighbors)) {
        return false;
    }

    runtime.buildSimSpaceMapping();
    if (!runtime.buildSimBuffers(memoryAllocator, commandPool)) {
        return false;
    }

    return rebuildOccupancyPointBuffer(runtime);
}

bool VoronoiSystemBuildStage::stageSurfaceMappings(VoronoiSystemRuntime& runtime) const {
    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime || domainRuntime->isPointDomain() || !runtime.getIntegrator()) {
        return false;
    }

    auto* modelRuntime = static_cast<VoronoiModelRuntime*>(domainRuntime);

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
    const std::vector<uint32_t>& voronoiToSim = runtime.getVoronoiToSim();

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
            const uint32_t voronoiNodeId = sourceGlobalIndices[neighborIndex];
            if (voronoiNodeId >= voronoiToSim.size()) {
                continue;
            }
            const uint32_t simNodeId = voronoiToSim[voronoiNodeId];
            if (simNodeId == UINT32_MAX) {
                continue;
            }
            const float valueWeight = static_cast<float>(valueWeightDoubles[neighborIndex]);
            const glm::dvec3 gradientWeight = gradientWeightTriples[neighborIndex];

            if (std::abs(valueWeight) > 1e-7f) {
                valueWeights.push_back({ simNodeId, valueWeight });
                ++stencil.valueWeightCount;
            }

            if (glm::dot(gradientWeight, gradientWeight) > 1e-14) {
                gradientWeights.push_back({
                    simNodeId,
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


void VoronoiSystemBuildStage::cleanupResources() {
    if (voronoiGeoCompute) {
        voronoiGeoCompute->cleanupResources();
    }
}
