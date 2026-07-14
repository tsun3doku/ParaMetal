#include "VoronoiSystemRuntime.hpp"

#include <algorithm>
#include <cmath>

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiCandidateCompute.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"
#include "voronoi/VoronoiPointRuntime.hpp"

#include <iostream>
#include <limits>

void VoronoiSystemRuntime::invalidateMaterialization() {
    voronoiReady = false;
    voronoiMeshGridReady = false;
    meshGrid.reset();
    integrator.reset();
    voxelGrid = VoxelGrid{};
    voxelGridBuilt = false;
    seedFlags.clear();
    seedPositions.clear();
    neighborIndices.clear();
    meshTriangles.clear();
    nodeDomain = {};
}

void VoronoiSystemRuntime::reorderNodes() {
    if (seedPositions.empty()) {
        return;
    }

    auto oldToNew = computeMortonPermutation(seedPositions);
    if (oldToNew.empty()) {
        return;
    }

    const uint32_t nodeCount = static_cast<uint32_t>(seedPositions.size());

    // Reorder seedPositions
    std::vector<glm::vec4> newPositions(nodeCount);
    for (uint32_t oldIdx = 0; oldIdx < nodeCount; ++oldIdx) {
        newPositions[oldToNew[oldIdx]] = seedPositions[oldIdx];
    }
    seedPositions = std::move(newPositions);

    // Reorder seedFlags
    if (!seedFlags.empty() && seedFlags.size() == nodeCount) {
        std::vector<uint32_t> newFlags(nodeCount);
        for (uint32_t oldIdx = 0; oldIdx < nodeCount; ++oldIdx) {
            newFlags[oldToNew[oldIdx]] = seedFlags[oldIdx];
        }
        seedFlags = std::move(newFlags);
    }

    // Remap neighbor indices and reorder the flat array
    if (!neighborIndices.empty()) {
        const uint32_t maxNeighbors = static_cast<uint32_t>(neighborIndices.size()) / nodeCount;
        std::vector<uint32_t> newNeighbors(neighborIndices.size());
        for (uint32_t oldIdx = 0; oldIdx < nodeCount; ++oldIdx) {
            const uint32_t newIdx = oldToNew[oldIdx];
            for (uint32_t k = 0; k < maxNeighbors; ++k) {
                uint32_t neighbor = neighborIndices[oldIdx * maxNeighbors + k];
                if (neighbor != UINT32_MAX && neighbor < nodeCount) {
                    neighbor = oldToNew[neighbor];
                }
                newNeighbors[newIdx * maxNeighbors + k] = neighbor;
            }
        }
        neighborIndices = std::move(newNeighbors);
    }
}

void VoronoiSystemRuntime::setMeshGeometry(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool,
    const std::vector<glm::vec3>& geometryPositions,
    const std::vector<uint32_t>& geometryTriangleIndices,
    const std::vector<voronoi::SurfaceVertex>& surfaceVertices,
    const std::vector<uint32_t>& surfaceTriangleIndices,
    uint32_t runtimeModelId,
    const glm::mat4& meshModelMatrix) {
    invalidateMaterialization();

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();

    if (runtimeModelId == 0) {
        return;
    }

    auto nextModelRuntime = std::make_unique<VoronoiModelRuntime>(
        vulkanDevice,
        memoryAllocator,
        runtimeModelId,
        meshModelMatrix,
        geometryPositions,
        geometryTriangleIndices,
        surfaceVertices,
        surfaceTriangleIndices,
        renderCommandPool);
    if (!nextModelRuntime->createVoronoiBuffers()) {
        std::cerr << "[VoronoiSystemRuntime] Failed to create Voronoi buffers for runtimeModelId="
                  << runtimeModelId << std::endl;
        nextModelRuntime->cleanup();
        return;
    }

    if (!nextModelRuntime->createSurfaceBuffers()) {
        std::cerr << "[VoronoiSystemRuntime] Failed to create surface buffers for runtimeModelId="
                  << runtimeModelId << std::endl;
        nextModelRuntime->cleanup();
        return;
    }

    domainRuntime = std::move(nextModelRuntime);
}

void VoronoiSystemRuntime::setPointGeometry(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool,
    uint64_t domainKey,
    const std::vector<glm::vec4>& positions) {
    invalidateMaterialization();

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();

    if (positions.empty()) {
        return;
    }

    auto nextPointRuntime = std::make_unique<VoronoiPointRuntime>(
        vulkanDevice,
        memoryAllocator,
        domainKey,
        positions,
        std::vector<uint32_t>{},    // no boundary conditions for raw points
        std::vector<float>{},       // no fixed temperatures for raw points
        renderCommandPool);
    if (!nextPointRuntime->createVoronoiBuffers()) {
        std::cerr << "[VoronoiSystemRuntime] Failed to create Voronoi buffers for point domain key="
                  << domainKey << std::endl;
        nextPointRuntime->cleanup();
        return;
    }

    domainRuntime = std::move(nextPointRuntime);
    setSeedPositions(positions);
}

void VoronoiSystemRuntime::setSeedPositions(const std::vector<glm::vec4>& positions) {
    seedPositions = positions;
    seedFlags.assign(positions.size(), 0u);
}

void VoronoiSystemRuntime::clearGeometry() {
    invalidateMaterialization();

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();
}

void VoronoiSystemRuntime::setParams(float updatedCellSize, int updatedVoxelResolution) {
    if (cellSize == updatedCellSize && voxelResolution == updatedVoxelResolution) {
        return;
    }

    cellSize = updatedCellSize;
    voxelResolution = updatedVoxelResolution;
    invalidateMaterialization();
}

void VoronoiSystemRuntime::markMeshGridReady() {
    voronoiMeshGridReady = true;
}

void VoronoiSystemRuntime::markReady() {
    voronoiReady = true;
}

void VoronoiSystemRuntime::cleanup(MemoryAllocator& memoryAllocator) {
    invalidateMaterialization();

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();
}
