#include "VoronoiSystemBuildStage.hpp"

#include "spatial/VoxelGrid.hpp"
#include "voronoi/VoronoiDomainRuntime.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "voronoi/VoronoiSystemRuntime.hpp"

#include "renderers/PointRenderer.hpp"
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

VoronoiSystemBuildStage::VoronoiSystemBuildStage(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
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
    const uint32_t runtimeModelId = modelRuntime->getRuntimeModelId();
    if (runtimeModelId == 0) {
        return false;
    }

    runtime.resetMeshGrid();
    if (!runtime.getMeshGrid()) {
        return false;
    }

    const std::vector<glm::vec3>& geometryPositions = modelRuntime->getGeometryPositions();
    const std::vector<uint32_t>& geometryIndices = modelRuntime->getGeometryTriangleIndices();
    runtime.getMeshGrid()->buildGrids(
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
    if (candidateNodeCount == 0 || !nodeFlagsBuffer) {
        return;
    }

    std::vector<voronoi::Node> nodes(candidateNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool,
        candidateNodeBuffer, candidateNodeBufferOffset,
        nodes.size() * sizeof(voronoi::Node), nodes.data()) != VK_SUCCESS) {
        return;
    }

    uint32_t promotedCount = 0;
    for (uint32_t i = 0; i < candidateNodeCount; ++i) {
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

bool VoronoiSystemBuildStage::createCandidateBuffers(const std::vector<voronoi::Node> &candidateNodes,
                                                     const std::vector<glm::vec4> &seedPositions,
                                                     const std::vector<uint32_t> &neighborIndices) {
    if (seedPositions.empty() || candidateNodes.empty() || neighborIndices.empty()) {
        return false;
    }

    if (seedPositions.size() != candidateNodes.size()) {
        return false;
    }

    candidateNodeCount = static_cast<uint32_t>(seedPositions.size());

    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (uploadDeviceBuffer(memoryAllocator, commandPool, candidateNodes.data(),
                           sizeof(voronoi::Node) * candidateNodeCount,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           candidateNodeBuffer, candidateNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(memoryAllocator, commandPool, neighborIndices.data(),
                           sizeof(uint32_t) * neighborIndices.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           storageAlignment, candidateNeighborIndicesBuffer,
                           candidateNeighborIndicesBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(memoryAllocator, commandPool, seedPositions.data(), sizeof(glm::vec4) * seedPositions.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           seedPositionBuffer, seedPositionBufferOffset) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VoronoiSystemBuildStage::createMeshGeometryBuffers(const std::vector<uint32_t> &nodeFlags, bool debugEnabled,
                                                        uint32_t maxNeighbors) {
    if (nodeFlags.size() != candidateNodeCount || candidateNodeCount == 0) {
        return false;
    }

    const VkDeviceSize storageAlignment =
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    const size_t interfaceDataSize = static_cast<size_t>(candidateNodeCount) * maxNeighbors;

    std::vector<float> emptySurfacePatchAreas(candidateNodeCount, 0.0f);
    if (uploadDeviceBuffer(memoryAllocator, commandPool, emptySurfacePatchAreas.data(),
                           sizeof(float) * emptySurfacePatchAreas.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           surfacePatchAreasBuffer, surfacePatchAreasBufferOffset) != VK_SUCCESS) {
        return false;
    }

    std::vector<float> emptyAreas(interfaceDataSize, 0.0f);
    if (uploadDeviceBuffer(memoryAllocator, commandPool, emptyAreas.data(), sizeof(float) * emptyAreas.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           candidateInterfaceAreasBuffer, candidateInterfaceAreasBufferOffset) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t> emptyIds(interfaceDataSize, UINT32_MAX);
    if (uploadDeviceBuffer(memoryAllocator, commandPool, emptyIds.data(), sizeof(uint32_t) * emptyIds.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           candidateInterfaceNeighborIdsBuffer,
                           candidateInterfaceNeighborIdsBufferOffset) != VK_SUCCESS) {
        return false;
    }

    const uint32_t debugCellCount = debugEnabled ? candidateNodeCount : 1u;
    std::vector<voronoi::DebugCellGeometry> debugCells(debugCellCount);
    if (uploadDeviceBuffer(memoryAllocator, commandPool, debugCells.data(),
                           sizeof(voronoi::DebugCellGeometry) * debugCells.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           debugCellGeometryBuffer, debugCellGeometryBufferOffset) != VK_SUCCESS) {
        return false;
    }

    const uint32_t dumpCount = debugEnabled ? voronoi::DEBUG_DUMP_CELL_COUNT : 1u;
    std::vector<voronoi::DumpInfo> dumpInfos(dumpCount);
    if (uploadDeviceBuffer(memoryAllocator, commandPool, dumpInfos.data(), sizeof(voronoi::DumpInfo) * dumpInfos.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           voronoiDumpBuffer, voronoiDumpBufferOffset) != VK_SUCCESS) {
        return false;
    }

    return uploadDeviceBuffer(memoryAllocator, commandPool, nodeFlags.data(), sizeof(uint32_t) * nodeFlags.size(),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                              nodeFlagsBuffer, nodeFlagsBufferOffset) == VK_SUCCESS;
}

bool VoronoiSystemBuildStage::buildMeshCouplingBuffer(VoronoiSystemRuntime &runtime, uint32_t maxNeighbors) {
    if (candidateNodeCount == 0) {
        return false;
    }

    std::vector<float> areas(static_cast<size_t>(candidateNodeCount) * maxNeighbors);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, candidateInterfaceAreasBuffer,
                             candidateInterfaceAreasBufferOffset, areas.size() * sizeof(float),
                             areas.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t> neighborIds(static_cast<size_t>(candidateNodeCount) * maxNeighbors);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, candidateInterfaceNeighborIdsBuffer,
                             candidateInterfaceNeighborIdsBufferOffset, neighborIds.size() * sizeof(uint32_t),
                             neighborIds.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<voronoi::Node> nodes(candidateNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, candidateNodeBuffer, candidateNodeBufferOffset,
        nodes.size() * sizeof(voronoi::Node), nodes.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<glm::vec4> seedPositions(candidateNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, seedPositionBuffer, seedPositionBufferOffset,
        seedPositions.size() * sizeof(glm::vec4), seedPositions.data()) != VK_SUCCESS) {
        return false;
    }

    const std::vector<uint32_t>& seedFlags = runtime.getSeedFlags();

    std::vector<voronoi::NodeCoupling> couplings;
    couplings.reserve(static_cast<size_t>(candidateNodeCount) * static_cast<size_t>(maxNeighbors));
    std::vector<voronoi::NodeCoupling> cellCouplings;
    cellCouplings.reserve(maxNeighbors);

    uint32_t totalNeighbors = 0;
    for (uint32_t cellIdx = 0; cellIdx < candidateNodeCount; ++cellIdx) {
        uint32_t neighborOffset = totalNeighbors;
        cellCouplings.clear();
        
        if ((seedFlags[cellIdx] & 1u) != 0u) {
            nodes[cellIdx].neighborOffset = neighborOffset;
            nodes[cellIdx].neighborCount = 0;
            continue;
        }

        uint32_t couplingCount = nodes[cellIdx].interfaceNeighborCount;
        if (couplingCount > maxNeighbors) {
            couplingCount = maxNeighbors;
        }

        const glm::vec3 cellPosition(seedPositions[cellIdx]);

        for (uint32_t k = 0; k < couplingCount; ++k) {
            const uint32_t idx = cellIdx * maxNeighbors + k;
            const uint32_t neighborNodeId = neighborIds[idx];
            const float area = areas[idx];
            
            if (neighborNodeId == UINT32_MAX || area <= 0.0f) {
                continue;
            }

            const glm::vec3 seedB(seedPositions[neighborNodeId]);
            const float distance = glm::length(seedB - cellPosition);
            if (distance <= 1e-12f || area <= 1e-8f) {
                continue;
            }

            const float conductance = area / distance;
            voronoi::NodeCoupling coupling{};
            coupling.neighborNodeId = neighborNodeId;
            coupling.conductance = conductance;

            cellCouplings.push_back(coupling);
        }

        std::sort(
            cellCouplings.begin(),
            cellCouplings.end(),
            [](const voronoi::NodeCoupling& a, const voronoi::NodeCoupling& b) {
                return a.neighborNodeId < b.neighborNodeId;
            });

        couplings.insert(couplings.end(), cellCouplings.begin(), cellCouplings.end());
        nodes[cellIdx].neighborOffset = neighborOffset;
        nodes[cellIdx].neighborCount = static_cast<uint32_t>(cellCouplings.size());
        totalNeighbors += static_cast<uint32_t>(cellCouplings.size());
    }

    if (couplings.empty()) {
        uint32_t ghostedCount = 0;
        uint32_t zeroInterfaceCount = 0;
        for (uint32_t i = 0; i < candidateNodeCount; ++i) {
            if ((seedFlags[i] & 1u) != 0u) {
                ++ghostedCount;
            } else if (nodes[i].interfaceNeighborCount == 0) {
                ++zeroInterfaceCount;
            }
        }
        return false;
    }

    candidateCouplingBuffer = VK_NULL_HANDLE;
    candidateCouplingBufferOffset = 0;
    VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    
    if (uploadDeviceBuffer(memoryAllocator, commandPool, couplings.data(),
        sizeof(voronoi::NodeCoupling) * couplings.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment, 
        candidateCouplingBuffer, candidateCouplingBufferOffset) != VK_SUCCESS) {
        return false;
    }

    candidateNodeBuffer = VK_NULL_HANDLE;
    candidateNodeBufferOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, nodes.data(),
        sizeof(voronoi::Node) * nodes.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
        candidateNodeBuffer, candidateNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }

    memoryAllocator.free(candidateInterfaceAreasBuffer, candidateInterfaceAreasBufferOffset);
    candidateInterfaceAreasBuffer = VK_NULL_HANDLE;
    candidateInterfaceAreasBufferOffset = 0;
    memoryAllocator.free(candidateInterfaceNeighborIdsBuffer, candidateInterfaceNeighborIdsBufferOffset);
    candidateInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
    candidateInterfaceNeighborIdsBufferOffset = 0;

    return true;
}

bool VoronoiSystemBuildStage::rebuildOccupancyPointBuffer(VoronoiSystemRuntime& runtime) {
    if (occupancyPointBuffer != VK_NULL_HANDLE) {
        occupancyPointBuffer = VK_NULL_HANDLE;
        occupancyPointBufferOffset = 0;
    }
    occupancyPointCount = 0;

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
    occupancyPointBuffer = buffer;
    occupancyPointBufferOffset = offset;
    occupancyPointCount = static_cast<uint32_t>(points.size());
    return true;
}

bool VoronoiSystemBuildStage::buildPointTopology(VoronoiSystemRuntime &runtime,
                                                 std::vector<voronoi::Node> &candidateNodes,
                                                 const std::vector<glm::vec4> &candidatePositions,
                                                 const std::vector<uint32_t> &neighborIndices, uint32_t maxNeighbors) {
    std::vector<voronoi::NodeCoupling> couplings;
    if (!runtime.getIntegrator()->buildPointCouplings(candidatePositions, runtime.getSeedFlags(), neighborIndices,
                                                       maxNeighbors, candidateNodes, couplings)) {
        return false;
    }

    const VkDeviceSize storageAlignment =
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, couplings.data(),
                           sizeof(voronoi::NodeCoupling) * couplings.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           candidateCouplingBuffer, candidateCouplingBufferOffset) != VK_SUCCESS) {
        return false;
    }

    candidateNodeBuffer = VK_NULL_HANDLE;
    candidateNodeBufferOffset = 0;
    return uploadDeviceBuffer(memoryAllocator, commandPool, candidateNodes.data(),
                              sizeof(voronoi::Node) * candidateNodes.size(),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                              candidateNodeBuffer, candidateNodeBufferOffset) == VK_SUCCESS;
}

bool VoronoiSystemBuildStage::buildMeshTopology(VoronoiSystemRuntime &runtime, const std::vector<uint32_t> &candidateFlags, bool debugEnabled, uint32_t maxNeighbors) {
    if (!createMeshGeometryBuffers(candidateFlags, debugEnabled, maxNeighbors)) {
        return false;
    }

    if (voronoiGeoCompute) {
        voronoiGeoCompute->initialize(candidateNodeCount);
    }

    freeBuffer(memoryAllocator, meshTriangleBuffer, meshTriangleBufferOffset);
    freeBuffer(memoryAllocator, voxelGridParamsBuffer, voxelGridParamsBufferOffset);
    freeBuffer(memoryAllocator, voxelOccupancyBuffer, voxelOccupancyBufferOffset);
    freeBuffer(memoryAllocator, voxelTrianglesListBuffer, voxelTrianglesListBufferOffset);
    freeBuffer(memoryAllocator, voxelOffsetsBuffer, voxelOffsetsBufferOffset);

    const auto &meshTris = runtime.getMeshTriangles();
    if (meshTris.empty()) {
        return false;
    }

    const VkDeviceSize storageAlignment =
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (uploadDeviceBuffer(memoryAllocator, commandPool, meshTris.data(), sizeof(glm::vec4) * 3 * meshTris.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, storageAlignment, meshTriangleBuffer,
                           meshTriangleBufferOffset) != VK_SUCCESS) {
        return false;
    }

    VoxelGrid::VoxelGridParams params{};
    std::vector<uint32_t> occupancy32;
    std::vector<int32_t> trianglesList;
    std::vector<int32_t> offsets;

    if (runtime.isVoxelGridBuilt()) {
        params = runtime.getVoxelGrid().getParams();
        const auto &occupancy8 = runtime.getVoxelGrid().getOccupancyData();
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
        void *mapped = nullptr;
        if (createUniformBuffer(memoryAllocator, vulkanDevice, sizeof(VoxelGrid::VoxelGridParams),
                                voxelGridParamsBuffer, voxelGridParamsBufferOffset, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(mapped, &params, sizeof(VoxelGrid::VoxelGridParams));
    }

    if (uploadDeviceBuffer(memoryAllocator, commandPool, occupancy32.data(), sizeof(uint32_t) * occupancy32.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, storageAlignment, voxelOccupancyBuffer,
                           voxelOccupancyBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(memoryAllocator, commandPool, trianglesList.data(),
                           sizeof(int32_t) * trianglesList.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           storageAlignment, voxelTrianglesListBuffer,
                           voxelTrianglesListBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (uploadDeviceBuffer(memoryAllocator, commandPool, offsets.data(), sizeof(int32_t) * offsets.size(),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, storageAlignment, voxelOffsetsBuffer,
                           voxelOffsetsBufferOffset) != VK_SUCCESS) {
        return false;
    }

    if (!voronoiGeoCompute) {
        return false;
    }

    VoronoiGeoCompute::Bindings bindings{};
    bindings.candidateNodeBuffer = candidateNodeBuffer;
    bindings.candidateNodeBufferOffset = candidateNodeBufferOffset;
    bindings.candidateNodeBufferRange = sizeof(voronoi::Node) * candidateNodeCount;
    bindings.meshTriangleBuffer = meshTriangleBuffer;
    bindings.meshTriangleBufferOffset = meshTriangleBufferOffset;
    bindings.seedPositionBuffer = seedPositionBuffer;
    bindings.seedPositionBufferOffset = seedPositionBufferOffset;
    bindings.voxelGridParamsBuffer = voxelGridParamsBuffer;
    bindings.voxelGridParamsBufferOffset = voxelGridParamsBufferOffset;
    bindings.voxelGridParamsBufferRange = sizeof(VoxelGrid::VoxelGridParams);
    bindings.voxelOccupancyBuffer = voxelOccupancyBuffer;
    bindings.voxelOccupancyBufferOffset = voxelOccupancyBufferOffset;
    bindings.voxelTrianglesListBuffer = voxelTrianglesListBuffer;
    bindings.voxelTrianglesListBufferOffset = voxelTrianglesListBufferOffset;
    bindings.voxelOffsetsBuffer = voxelOffsetsBuffer;
    bindings.voxelOffsetsBufferOffset = voxelOffsetsBufferOffset;
    bindings.candidateNeighborIndicesBuffer = candidateNeighborIndicesBuffer;
    bindings.candidateNeighborIndicesBufferOffset = candidateNeighborIndicesBufferOffset;
    bindings.candidateInterfaceAreasBuffer = candidateInterfaceAreasBuffer;
    bindings.candidateInterfaceAreasBufferOffset = candidateInterfaceAreasBufferOffset;
    bindings.candidateInterfaceNeighborIdsBuffer = candidateInterfaceNeighborIdsBuffer;
    bindings.candidateInterfaceNeighborIdsBufferOffset = candidateInterfaceNeighborIdsBufferOffset;
    bindings.debugCellGeometryBuffer = debugCellGeometryBuffer;
    bindings.debugCellGeometryBufferOffset = debugCellGeometryBufferOffset;
    bindings.nodeFlagsBuffer = nodeFlagsBuffer;
    bindings.nodeFlagsBufferOffset = nodeFlagsBufferOffset;
    bindings.voronoiDumpBuffer = voronoiDumpBuffer;
    bindings.voronoiDumpBufferOffset = voronoiDumpBufferOffset;
    bindings.surfacePatchAreasBuffer = surfacePatchAreasBuffer;
    bindings.surfacePatchAreasBufferOffset = surfacePatchAreasBufferOffset;

    voronoiGeoCompute->updateDescriptors(bindings);

    VoronoiGeoCompute::PushConstants geoPushConstants{};
    geoPushConstants.debugEnable = debugEnabled ? 1u : 0u;
    geoPushConstants.nodeOffset = 0;
    geoPushConstants.nodeCount = candidateNodeCount;
    if (!voronoiGeoCompute->dispatch(geoPushConstants)) {
        return false;
    }

    freeBuffer(memoryAllocator, meshTriangleBuffer, meshTriangleBufferOffset);
    freeBuffer(memoryAllocator, voxelGridParamsBuffer, voxelGridParamsBufferOffset);
    freeBuffer(memoryAllocator, voxelOccupancyBuffer, voxelOccupancyBufferOffset);
    freeBuffer(memoryAllocator, voxelTrianglesListBuffer, voxelTrianglesListBufferOffset);
    freeBuffer(memoryAllocator, voxelOffsetsBuffer, voxelOffsetsBufferOffset);

    setGhostFromVolumes(runtime);
    if (!readSurfaceData(runtime)) {
        return false;
    }
    if (!buildMeshCouplingBuffer(runtime, maxNeighbors)) {
        return false;
    }
    cleanupMeshGeometryBuffers();
    return true;
}

bool VoronoiSystemBuildStage::dispatchVoronoiCompute(VoronoiSystemRuntime &runtime, bool debugEnabled,
                                                     uint32_t maxNeighbors) {
    VoronoiDomainRuntime *domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime || !runtime.getIntegrator()) {
        return false;
    }

    const auto &seedPositions = runtime.getSeedPositions();
    if (seedPositions.empty()) {
        return false;
    }
    if (runtime.getSeedFlags().size() != seedPositions.size()) {
        return false;
    }

    candidateNodeCount = static_cast<uint32_t>(seedPositions.size());
    if (candidateNodeCount == 0) {
        return false;
    }

    std::vector<voronoi::Node> candidateNodes(candidateNodeCount);
    for (voronoi::Node &node : candidateNodes) {
        node.volume = 0.0f;
        node.neighborOffset = 0u;
        node.neighborCount = 0u;
        node.interfaceNeighborCount = 0u;
    }

    std::vector<glm::vec4> candidatePositions;
    std::vector<uint32_t> candidateFlags;
    std::vector<uint32_t> candidateNeighborIndices;
    candidatePositions.reserve(candidateNodeCount);
    candidateFlags.reserve(candidateNodeCount);
    candidateNeighborIndices.reserve(static_cast<size_t>(candidateNodeCount) * static_cast<size_t>(maxNeighbors));

    const auto &neighborIndices = runtime.getNeighborIndices();
    const size_t expectedNeighborCount = static_cast<size_t>(candidateNodeCount) * static_cast<size_t>(maxNeighbors);
    if (neighborIndices.size() < expectedNeighborCount) {
        return false;
    }

    candidatePositions.insert(candidatePositions.end(), seedPositions.begin(), seedPositions.end());
    candidateFlags.insert(candidateFlags.end(), runtime.getSeedFlags().begin(), runtime.getSeedFlags().end());

    for (uint32_t localNodeIndex = 0; localNodeIndex < candidateNodeCount; ++localNodeIndex) {
        const size_t base = static_cast<size_t>(localNodeIndex) * static_cast<size_t>(maxNeighbors);
        for (uint32_t k = 0; k < maxNeighbors; ++k) {
            uint32_t neighborIndex = neighborIndices[base + static_cast<size_t>(k)];
            if (neighborIndex != UINT32_MAX) {
                if (neighborIndex >= candidateNodeCount) {
                    neighborIndex = UINT32_MAX;
                }
            }
            candidateNeighborIndices.push_back(neighborIndex);
        }
    }

    if (candidatePositions.size() != candidateNodeCount || candidateFlags.size() != candidateNodeCount) {
        return false;
    }

    if (!createCandidateBuffers(candidateNodes, candidatePositions, candidateNeighborIndices)) {
        return false;
    }

    if (domainRuntime->isPointDomain()) {
        candidateSurfacePatchAreas.assign(candidateNodeCount, 0.0f);
        if (!buildPointTopology(runtime, candidateNodes, candidatePositions, candidateNeighborIndices, maxNeighbors)) {
            return false;
        }
    }

    if (!domainRuntime->isPointDomain() && !buildMeshTopology(runtime, candidateFlags, debugEnabled, maxNeighbors)) {
        return false;
    }

    if (!finalizeNodeDomain(runtime)) {
        return false;
    }

    cleanupCandidateTopologyBuffers();

    const VoronoiNodeDomain &nodeDomain = runtime.getNodeDomain();
    if (nodeDomain.getCouplings().empty()) {
        return false;
    }

    return domainRuntime->isPointDomain() || rebuildOccupancyPointBuffer(runtime);
}

bool VoronoiSystemBuildStage::finalizeNodeDomain(VoronoiSystemRuntime &runtime) {
    if (candidateNodeCount == 0 || candidateNodeBuffer == VK_NULL_HANDLE ||
        candidateCouplingBuffer == VK_NULL_HANDLE) {
        return false;
    }

    std::vector<voronoi::Node> candidateNodes(candidateNodeCount);
    if (downloadDeviceBuffer(memoryAllocator, commandPool, candidateNodeBuffer, candidateNodeBufferOffset,
                             candidateNodes.size() * sizeof(voronoi::Node), candidateNodes.data()) != VK_SUCCESS) {
        return false;
    }

    size_t candidateCouplingCount = 0;
    for (const voronoi::Node &node : candidateNodes) {
        candidateCouplingCount =
            std::max(candidateCouplingCount, static_cast<size_t>(node.neighborOffset) + node.neighborCount);
    }
    if (candidateCouplingCount == 0) {
        return false;
    }

    std::vector<voronoi::NodeCoupling> candidateCouplings(candidateCouplingCount);
    if (downloadDeviceBuffer(
            memoryAllocator, commandPool, candidateCouplingBuffer, candidateCouplingBufferOffset,
            candidateCouplings.size() * sizeof(voronoi::NodeCoupling), candidateCouplings.data()) != VK_SUCCESS) {
        return false;
    }

    runtime.getNodeDomain().rebuild(runtime.getSeedFlags(), runtime.getSeedPositions(), candidateNodes,
                                    candidateCouplings, candidateSurfacePatchAreas);
    const VoronoiNodeDomain& nodeDomain = runtime.getNodeDomain();
    return nodeDomain.getNodeCount() != 0 && uploadNodeDomainBuffers(nodeDomain);
}

bool VoronoiSystemBuildStage::uploadNodeDomainBuffers(const VoronoiNodeDomain &nodeDomain) {
    if (nodeDomain.getNodes().empty() || nodeDomain.getCouplings().empty()) {
        return false;
    }

    constexpr VkDeviceSize storageAlignment = 16;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, nodeDomain.getNodes().data(),
                           nodeDomain.getNodes().size() * sizeof(voronoi::Node),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           nodeBuffer, nodeBufferOffset) != VK_SUCCESS) {
        return false;
    }
    if (uploadDeviceBuffer(memoryAllocator, commandPool, nodeDomain.getCouplings().data(),
                           nodeDomain.getCouplings().size() * sizeof(voronoi::NodeCoupling),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, storageAlignment,
                           couplingBuffer, couplingBufferOffset) != VK_SUCCESS) {
        return false;
    }
    couplingCount = static_cast<uint32_t>(nodeDomain.getCouplings().size());
    return true;
}

void VoronoiSystemBuildStage::cleanupCandidateTopologyBuffers() {
    memoryAllocator.free(candidateCouplingBuffer, candidateCouplingBufferOffset);
    candidateCouplingBuffer = VK_NULL_HANDLE;
    candidateCouplingBufferOffset = 0;
    memoryAllocator.free(nodeFlagsBuffer, nodeFlagsBufferOffset);
    nodeFlagsBuffer = VK_NULL_HANDLE;
    nodeFlagsBufferOffset = 0;
}

void VoronoiSystemBuildStage::cleanupMeshGeometryBuffers() {
    freeBuffer(memoryAllocator, candidateInterfaceAreasBuffer, candidateInterfaceAreasBufferOffset);
    freeBuffer(memoryAllocator, candidateInterfaceNeighborIdsBuffer, candidateInterfaceNeighborIdsBufferOffset);
    freeBuffer(memoryAllocator, meshTriangleBuffer, meshTriangleBufferOffset);
    freeBuffer(memoryAllocator, voxelGridParamsBuffer, voxelGridParamsBufferOffset);
    freeBuffer(memoryAllocator, voxelOccupancyBuffer, voxelOccupancyBufferOffset);
    freeBuffer(memoryAllocator, voxelTrianglesListBuffer, voxelTrianglesListBufferOffset);
    freeBuffer(memoryAllocator, voxelOffsetsBuffer, voxelOffsetsBufferOffset);
    freeBuffer(memoryAllocator, debugCellGeometryBuffer, debugCellGeometryBufferOffset);
    freeBuffer(memoryAllocator, voronoiDumpBuffer, voronoiDumpBufferOffset);
    freeBuffer(memoryAllocator, surfacePatchAreasBuffer, surfacePatchAreasBufferOffset);
}

bool VoronoiSystemBuildStage::readSurfaceData(VoronoiSystemRuntime& runtime) {
    if (candidateNodeCount == 0 || nodeFlagsBuffer == VK_NULL_HANDLE || surfacePatchAreasBuffer == VK_NULL_HANDLE) {
        return false;
    }

    std::vector<uint32_t> gpuFlags(candidateNodeCount, 0u);
    if (downloadDeviceBuffer(
            memoryAllocator,
            commandPool,
        nodeFlagsBuffer,
        nodeFlagsBufferOffset,
            gpuFlags.size() * sizeof(uint32_t),
            gpuFlags.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<uint32_t>& nodeFlags = runtime.getSeedFlags();
    if (nodeFlags.size() != gpuFlags.size()) {
        return false;
    }

    for (uint32_t nodeId = 0; nodeId < candidateNodeCount; ++nodeId) {
        nodeFlags[nodeId] = (nodeFlags[nodeId] & ~voronoi::NodeFlags::Surface) |
            (gpuFlags[nodeId] & voronoi::NodeFlags::Surface);
    }

    candidateSurfacePatchAreas.assign(candidateNodeCount, 0.0f);
    if (downloadDeviceBuffer(
            memoryAllocator,
            commandPool,
            surfacePatchAreasBuffer,
            surfacePatchAreasBufferOffset,
            candidateSurfacePatchAreas.size() * sizeof(float),
            candidateSurfacePatchAreas.data()) != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool VoronoiSystemBuildStage::stageSurfaceMappings(VoronoiSystemRuntime& runtime) const {
    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime || domainRuntime->isPointDomain() || !runtime.getIntegrator()) {
        return false;
    }

    auto* modelRuntime = static_cast<VoronoiModelRuntime*>(domainRuntime);
    std::vector<glm::vec3> surfacePoints = modelRuntime->getSurfacePositions();
    const std::vector<voronoi::SurfaceVertex>& surfaceVertices = modelRuntime->getSurfaceVertices();
    if (surfacePoints.empty() || surfaceVertices.size() != surfacePoints.size()) {
        return false;
    }

    std::vector<glm::vec3> surfaceNormals(surfaceVertices.size());
    for (size_t vertexId = 0; vertexId < surfaceVertices.size(); ++vertexId) {
        surfaceNormals[vertexId] = glm::vec3(surfaceVertices[vertexId].normal);
    }
    if (!runtime.getNodeDomain().buildSurfaceMappings(
            surfacePoints,
            surfaceNormals,
            runtime.getVoxelGrid())) {
        return false;
    }
    const VoronoiNodeDomain& nodeDomain = runtime.getNodeDomain();
    modelRuntime->stageGMLSSurfaceData(
        nodeDomain.getSurfaceStencils(),
        nodeDomain.getSurfaceValueWeights(),
        nodeDomain.getSurfaceGradientWeights());
    return true;
}


void VoronoiSystemBuildStage::cleanupResources() {
    if (voronoiGeoCompute) {
        voronoiGeoCompute->cleanupResources();
    }
}

void VoronoiSystemBuildStage::cleanup() {
    auto freeBuffer = [&](VkBuffer& buffer, VkDeviceSize& offset) {
        if (buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(buffer, offset);
            buffer = VK_NULL_HANDLE;
            offset = 0;
        }
    };
    auto dropHandle = [](VkBuffer& buffer, VkDeviceSize& offset) {
        buffer = VK_NULL_HANDLE;
        offset = 0;
    };

    dropHandle(candidateNodeBuffer, candidateNodeBufferOffset);
    dropHandle(candidateNeighborIndicesBuffer, candidateNeighborIndicesBufferOffset);
    dropHandle(seedPositionBuffer, seedPositionBufferOffset);
    dropHandle(occupancyPointBuffer, occupancyPointBufferOffset);
    dropHandle(nodeBuffer, nodeBufferOffset);
    dropHandle(couplingBuffer, couplingBufferOffset);
    candidateNodeCount = 0;
    occupancyPointCount = 0;
    couplingCount = 0;

    freeBuffer(candidateCouplingBuffer, candidateCouplingBufferOffset);
    freeBuffer(nodeFlagsBuffer, nodeFlagsBufferOffset);
    freeBuffer(surfacePatchAreasBuffer, surfacePatchAreasBufferOffset);
    candidateSurfacePatchAreas.clear();
    cleanupMeshGeometryBuffers();
}
