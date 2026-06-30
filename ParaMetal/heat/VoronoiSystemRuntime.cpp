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
    voronoiToSim.clear();
    simToVoronoi.clear();
    simNodeVolumes.clear();
    simNodeCount = 0;
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

void VoronoiSystemRuntime::buildSimSpaceMapping() {
    const uint32_t voronoiNodeCount = static_cast<uint32_t>(seedFlags.size());
    voronoiToSim.assign(voronoiNodeCount, UINT32_MAX);
    simToVoronoi.clear();
    simToVoronoi.reserve(voronoiNodeCount);

    for (uint32_t voronoiNodeId = 0; voronoiNodeId < voronoiNodeCount; ++voronoiNodeId) {
        if ((seedFlags[voronoiNodeId] & 1u) != 0u) {
            continue;
        }

        const uint32_t simNodeId = static_cast<uint32_t>(simToVoronoi.size());
        voronoiToSim[voronoiNodeId] = simNodeId;
        simToVoronoi.push_back(voronoiNodeId);
    }

    simNodeCount = static_cast<uint32_t>(simToVoronoi.size());
}

bool VoronoiSystemRuntime::buildSimBuffers(MemoryAllocator& memoryAllocator, CommandPool& renderCommandPool) {
    if (resources.voronoiNodeCount == 0 ||
        resources.voronoiNodeBuffer == VK_NULL_HANDLE ||
        resources.voronoiGMLSInterfaceBuffer == VK_NULL_HANDLE) {
        return false;
    }

    if (voronoiToSim.size() != resources.voronoiNodeCount || simToVoronoi.empty()) {
        buildSimSpaceMapping();
    }
    if (simNodeCount == 0) {
        return false;
    }

    std::vector<voronoi::Node> voronoiNodes(resources.voronoiNodeCount);
    if (downloadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            resources.voronoiNodeBuffer,
            resources.voronoiNodeBufferOffset,
            voronoiNodes.size() * sizeof(voronoi::Node),
            voronoiNodes.data()) != VK_SUCCESS) {
        return false;
    }

    size_t voronoiInterfaceCount = 0;
    for (const voronoi::Node& node : voronoiNodes) {
        voronoiInterfaceCount = std::max(
            voronoiInterfaceCount,
            static_cast<size_t>(node.neighborOffset) + static_cast<size_t>(node.neighborCount));
    }
    if (voronoiInterfaceCount == 0) {
        return false;
    }

    std::vector<voronoi::GMLSInterface> voronoiInterfaces(voronoiInterfaceCount);
    if (downloadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            resources.voronoiGMLSInterfaceBuffer,
            resources.voronoiGMLSInterfaceBufferOffset,
            voronoiInterfaces.size() * sizeof(voronoi::GMLSInterface),
            voronoiInterfaces.data()) != VK_SUCCESS) {
        return false;
    }

    std::vector<voronoi::Node> simNodes(simNodeCount);
    simNodeVolumes.assign(simNodeCount, 0.0f);
    std::vector<voronoi::GMLSInterface> simInterfaces;
    simInterfaces.reserve(voronoiInterfaces.size());

    for (uint32_t simNodeId = 0; simNodeId < simNodeCount; ++simNodeId) {
        const uint32_t voronoiNodeId = simToVoronoi[simNodeId];
        const voronoi::Node& voronoiNode = voronoiNodes[voronoiNodeId];

        voronoi::Node simNode{};
        simNode.volume = voronoiNode.volume;
        simNodeVolumes[simNodeId] = std::abs(voronoiNode.volume);
        simNode.interfaceNeighborCount = 0;
        simNode.neighborOffset = static_cast<uint32_t>(simInterfaces.size());

        const size_t interfaceBegin = voronoiNode.neighborOffset;
        const size_t interfaceEnd = interfaceBegin + voronoiNode.neighborCount;
        for (size_t interfaceIndex = interfaceBegin; interfaceIndex < interfaceEnd && interfaceIndex < voronoiInterfaces.size(); ++interfaceIndex) {
            const voronoi::GMLSInterface& voronoiInterface = voronoiInterfaces[interfaceIndex];
            if (voronoiInterface.neighborIdx >= voronoiToSim.size()) {
                continue;
            }

            const uint32_t neighborSimNodeId = voronoiToSim[voronoiInterface.neighborIdx];
            if (neighborSimNodeId == UINT32_MAX) {
                continue;
            }

            simInterfaces.push_back({neighborSimNodeId, voronoiInterface.conductance});
        }

        simNode.neighborCount = static_cast<uint32_t>(simInterfaces.size()) - simNode.neighborOffset;
        simNode.interfaceNeighborCount = simNode.neighborCount;
        simNodes[simNodeId] = simNode;
    }

    if (simInterfaces.empty()) {
        return false;
    }

    simNodeBuffer = VK_NULL_HANDLE;
    simNodeBufferOffset = 0;
    simNodeBufferSize = 0;
    simGMLSInterfaceBuffer = VK_NULL_HANDLE;
    simGMLSInterfaceBufferOffset = 0;
    simGMLSInterfaceCount = 0;

    const VkDeviceSize simNodeUploadSize = simNodes.size() * sizeof(voronoi::Node);

    constexpr VkDeviceSize storageAlignment = 16;
    if (uploadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            simNodes.data(),
            simNodeUploadSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            storageAlignment,
            simNodeBuffer,
            simNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }
    simNodeBufferSize = simNodeUploadSize;
    if (uploadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            simInterfaces.data(),
            simInterfaces.size() * sizeof(voronoi::GMLSInterface),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            simGMLSInterfaceBuffer,
            simGMLSInterfaceBufferOffset) != VK_SUCCESS) {
        return false;
    }

    simGMLSInterfaceCount = static_cast<uint32_t>(simInterfaces.size());
    return true;
}

void VoronoiSystemRuntime::setReceiverGeometry(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool,
    uint32_t receiverNodeModelId,
    const std::vector<glm::vec3>& receiverGeometryPositions,
    const std::vector<uint32_t>& receiverGeometryTriangleIndices,
    const SupportingHalfedge::IntrinsicMesh& receiverIntrinsicMesh,
    const std::vector<VoronoiModelRuntime::SurfaceVertex>& receiverSurfaceVertices,
    const std::vector<uint32_t>& receiverIntrinsicTriangleIndices,
    uint32_t receiverModelId,
    const glm::mat4& meshModelMatrix) {
    invalidateMaterialization();

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();

    if (receiverModelId == 0) {
        return;
    }

    auto nextModelRuntime = std::make_unique<VoronoiModelRuntime>(
        vulkanDevice,
        memoryAllocator,
        receiverModelId,
        meshModelMatrix,
        VoronoiModelRuntime::CpuData{
            receiverNodeModelId,
            receiverIntrinsicMesh,
            receiverGeometryPositions,
            receiverGeometryTriangleIndices,
            receiverSurfaceVertices,
            receiverIntrinsicTriangleIndices
        },
        renderCommandPool);
    if (!nextModelRuntime->createVoronoiBuffers()) {
        std::cerr << "[VoronoiSystemRuntime] Failed to create Voronoi buffers for runtimeModelId="
                  << receiverModelId << std::endl;
        nextModelRuntime->cleanup();
        return;
    }

    if (!nextModelRuntime->createSurfaceBuffers()) {
        std::cerr << "[VoronoiSystemRuntime] Failed to create surface buffers for runtimeModelId="
                  << receiverModelId << std::endl;
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
        std::vector<uint32_t>{},   // no boundary conditions for raw points
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

void VoronoiSystemRuntime::clearReceiverGeometry() {
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

void VoronoiSystemRuntime::cleanupResources(VulkanDevice& vulkanDevice) {
    if (resources.surfacePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), resources.surfacePipeline, nullptr);
        resources.surfacePipeline = VK_NULL_HANDLE;
    }
    if (resources.surfacePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), resources.surfacePipelineLayout, nullptr);
        resources.surfacePipelineLayout = VK_NULL_HANDLE;
    }
    if (resources.surfaceDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), resources.surfaceDescriptorPool, nullptr);
        resources.surfaceDescriptorPool = VK_NULL_HANDLE;
    }
    if (resources.surfaceDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), resources.surfaceDescriptorSetLayout, nullptr);
        resources.surfaceDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

void VoronoiSystemRuntime::cleanup(MemoryAllocator& memoryAllocator) {
    invalidateMaterialization();

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

    dropHandle(resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset);
    dropHandle(resources.voronoiNeighborBuffer, resources.voronoiNeighborBufferOffset);
    dropHandle(resources.voronoiNeighborIndicesBuffer, resources.voronoiNeighborIndicesBufferOffset);
    dropHandle(resources.voronoiInterfaceAreasBuffer, resources.voronoiInterfaceAreasBufferOffset);
    dropHandle(resources.voronoiInterfaceNeighborIdsBuffer, resources.voronoiInterfaceNeighborIdsBufferOffset);
    dropHandle(resources.voronoiGMLSInterfaceBuffer, resources.voronoiGMLSInterfaceBufferOffset);
    dropHandle(resources.seedPositionBuffer, resources.seedPositionBufferOffset);
    dropHandle(resources.voronoiSeedFlagsBuffer, resources.voronoiSeedFlagsBufferOffset);
    dropHandle(resources.occupancyPointBuffer, resources.occupancyPointBufferOffset);
    resources.occupancyPointCount = 0;
    dropHandle(simNodeBuffer, simNodeBufferOffset);
    simNodeBufferSize = 0;
    dropHandle(simGMLSInterfaceBuffer, simGMLSInterfaceBufferOffset);
    resources.voronoiNodeCount = 0;

    freeBuffer(resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(resources.debugCellGeometryBuffer, resources.debugCellGeometryBufferOffset);
    freeBuffer(resources.voronoiDumpBuffer, resources.voronoiDumpBufferOffset);
    freeBuffer(resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);

    if (domainRuntime) {
        domainRuntime->cleanup();
    }
    domainRuntime.reset();
}
