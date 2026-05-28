#include "VoronoiSystemRuntime.hpp"

#include <algorithm>

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiCandidateCompute.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"

#include <iostream>
#include <limits>

void VoronoiSystemRuntime::invalidateMaterialization() {
    voronoiReady = false;
    voronoiSeederReady = false;
    seeder.reset();
    integrator.reset();
    voxelGrid = VoxelGrid{};
    voxelGridBuilt = false;
    seedFlags.clear();
    seedPositions.clear();
    neighborIndices.clear();
    meshTriangles.clear();
    voronoiToSim.clear();
    simToVoronoi.clear();
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
    std::vector<voronoi::GMLSInterface> simInterfaces;
    simInterfaces.reserve(voronoiInterfaces.size());

    for (uint32_t simNodeId = 0; simNodeId < simNodeCount; ++simNodeId) {
        const uint32_t voronoiNodeId = simToVoronoi[simNodeId];
        const voronoi::Node& voronoiNode = voronoiNodes[voronoiNodeId];

        voronoi::Node simNode{};
        simNode.volume = voronoiNode.volume;
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

    freeBuffer(memoryAllocator, simNodeBuffer, simNodeBufferOffset);
    freeBuffer(memoryAllocator, simGMLSInterfaceBuffer, simGMLSInterfaceBufferOffset);

    constexpr VkDeviceSize storageAlignment = 16;
    if (uploadDeviceBuffer(
            memoryAllocator,
            renderCommandPool,
            simNodes.data(),
            simNodes.size() * sizeof(voronoi::Node),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            storageAlignment,
            simNodeBuffer,
            simNodeBufferOffset) != VK_SUCCESS) {
        return false;
    }

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

    if (modelRuntime) {
        modelRuntime->cleanup();
    }
    modelRuntime.reset();

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

    modelRuntime = std::move(nextModelRuntime);
}

void VoronoiSystemRuntime::clearReceiverGeometry() {
    invalidateMaterialization();

    if (modelRuntime) {
        modelRuntime->cleanup();
    }
    modelRuntime.reset();
}

void VoronoiSystemRuntime::setParams(float updatedCellSize, int updatedVoxelResolution) {
    if (cellSize == updatedCellSize && voxelResolution == updatedVoxelResolution) {
        return;
    }

    cellSize = updatedCellSize;
    voxelResolution = updatedVoxelResolution;
    invalidateMaterialization();
}

void VoronoiSystemRuntime::markSeederReady() {
    voronoiSeederReady = true;
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

    freeBuffer(resources.voronoiNodeBuffer, resources.voronoiNodeBufferOffset);
    freeBuffer(resources.voronoiNeighborBuffer, resources.voronoiNeighborBufferOffset);
    freeBuffer(resources.voronoiNeighborIndicesBuffer, resources.voronoiNeighborIndicesBufferOffset);
    freeBuffer(resources.voronoiInterfaceAreasBuffer, resources.voronoiInterfaceAreasBufferOffset);
    freeBuffer(resources.voronoiInterfaceNeighborIdsBuffer, resources.voronoiInterfaceNeighborIdsBufferOffset);
    freeBuffer(resources.voronoiGMLSInterfaceBuffer, resources.voronoiGMLSInterfaceBufferOffset);
    freeBuffer(resources.meshTriangleBuffer, resources.meshTriangleBufferOffset);
    freeBuffer(resources.seedPositionBuffer, resources.seedPositionBufferOffset);
    freeBuffer(resources.voronoiSeedFlagsBuffer, resources.voronoiSeedFlagsBufferOffset);
    freeBuffer(simNodeBuffer, simNodeBufferOffset);
    freeBuffer(simGMLSInterfaceBuffer, simGMLSInterfaceBufferOffset);
    freeBuffer(resources.occupancyPointBuffer, resources.occupancyPointBufferOffset);
    resources.occupancyPointCount = 0;
    freeBuffer(resources.debugCellGeometryBuffer, resources.debugCellGeometryBufferOffset);
    freeBuffer(resources.voronoiDumpBuffer, resources.voronoiDumpBufferOffset);
    freeBuffer(resources.voxelGridParamsBuffer, resources.voxelGridParamsBufferOffset);
    freeBuffer(resources.voxelOccupancyBuffer, resources.voxelOccupancyBufferOffset);
    freeBuffer(resources.voxelTrianglesListBuffer, resources.voxelTrianglesListBufferOffset);
    freeBuffer(resources.voxelOffsetsBuffer, resources.voxelOffsetsBufferOffset);
    resources.voronoiNodeCount = 0;

    if (modelRuntime) {
        modelRuntime->cleanup();
    }
    modelRuntime.reset();
}
