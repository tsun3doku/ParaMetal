#include "VoronoiModelRuntime.hpp"

#include "mesh/remesher/iODT.hpp"
#include "util/Structs.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiAdapters.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

VoronoiModelRuntime::VoronoiModelRuntime(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    uint32_t runtimeModelId,
    VkBuffer vertexBuffer,
    VkDeviceSize vertexBufferOffset,
    VkBuffer indexBuffer,
    VkDeviceSize indexBufferOffset,
    uint32_t indexCount,
    const glm::mat4& modelMatrix,
    CpuData cpuData,
    VkBufferView supportingHalfedgeView,
    VkBufferView supportingAngleView,
    VkBufferView halfedgeView,
    VkBufferView edgeView,
    VkBufferView triangleView,
    VkBufferView lengthView,
    VkBufferView inputHalfedgeView,
    VkBufferView inputEdgeView,
    VkBufferView inputTriangleView,
    VkBufferView inputLengthView,
    CommandPool& renderCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      nodeModelId(cpuData.nodeModelId),
      runtimeModelId(runtimeModelId),
      vertexBuffer(vertexBuffer),
      vertexBufferOffset(vertexBufferOffset),
      indexBuffer(indexBuffer),
      indexBufferOffset(indexBufferOffset),
      indexCount(indexCount),
      modelMatrix(modelMatrix),
      intrinsicMesh(std::move(cpuData.intrinsicMesh)),
      geometryPositions(std::move(cpuData.geometryPositions)),
      geometryTriangleIndices(std::move(cpuData.geometryTriangleIndices)),
      surfaceVertices(std::move(cpuData.surfaceVertices)),
      intrinsicTriangleIndices(std::move(cpuData.intrinsicTriangleIndices)),
      supportingHalfedgeView(supportingHalfedgeView),
      supportingAngleView(supportingAngleView),
      halfedgeView(halfedgeView),
      edgeView(edgeView),
      triangleView(triangleView),
      lengthView(lengthView),
      inputHalfedgeView(inputHalfedgeView),
      inputEdgeView(inputEdgeView),
      inputTriangleView(inputTriangleView),
      inputLengthView(inputLengthView),
      renderCommandPool(renderCommandPool) {
}

VoronoiModelRuntime::~VoronoiModelRuntime() {
}

bool VoronoiModelRuntime::createVoronoiBuffers() {
    if (intrinsicMesh.vertices.empty()) {
        std::cerr << "[VoronoiModelRuntime] Missing intrinsic state for model" << std::endl;
        return false;
    }

    const size_t vertexCount = intrinsicMesh.vertices.size();
    const size_t triangleCount = intrinsicMesh.indices.size() / 3;
    if (vertexCount == 0) {
        std::cerr << "[VoronoiModelRuntime] Model has 0 intrinsic vertices" << std::endl;
        return false;
    }

    intrinsicVertexCount = vertexCount;
    intrinsicTriangleCount = triangleCount;

    constexpr uint32_t K_CANDIDATES = 64;
    const VkDeviceSize triangleIndicesBufferSize = sizeof(uint32_t) * intrinsicTriangleIndices.size();
    const VkDeviceSize candidateBufferSize = sizeof(uint32_t) * triangleCount * K_CANDIDATES;
    const VkDeviceSize storageAlignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (triangleIndicesBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(triangleIndicesBuffer, triangleIndicesBufferOffset);
        triangleIndicesBuffer = VK_NULL_HANDLE;
        triangleIndicesBufferOffset = 0;
    }
    auto [triIdxHandle, triIdxOffset] = memoryAllocator.allocate(
        triangleIndicesBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        storageAlignment);
    if (triIdxHandle == VK_NULL_HANDLE) {
        std::cerr << "[VoronoiModelRuntime] Failed to allocate triangle index buffer" << std::endl;
        cleanup();
        return false;
    }
    triangleIndicesBuffer = triIdxHandle;
    triangleIndicesBufferOffset = triIdxOffset;

    VkBuffer triIdxStagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize triIdxStagingOffset = 0;
    void* triIdxStagingData = nullptr;
    if (createStagingBuffer(
            memoryAllocator,
            triangleIndicesBufferSize,
            triIdxStagingBuffer,
            triIdxStagingOffset,
            &triIdxStagingData) != VK_SUCCESS ||
        !triIdxStagingData) {
        std::cerr << "[VoronoiModelRuntime] Failed to create triangle index staging buffer" << std::endl;
        cleanup();
        return false;
    }
    std::memcpy(triIdxStagingData, intrinsicTriangleIndices.data(), static_cast<size_t>(triangleIndicesBufferSize));

    if (voronoiCandidateBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(voronoiCandidateBuffer, voronoiCandidateBufferOffset);
        voronoiCandidateBuffer = VK_NULL_HANDLE;
        voronoiCandidateBufferOffset = 0;
    }

    VkBuffer candidateStagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateStagingOffset = 0;
    if (candidateBufferSize > 0) {
        auto [candidateHandle, candidateOffset] = memoryAllocator.allocate(
            candidateBufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            storageAlignment);
        if (candidateHandle == VK_NULL_HANDLE) {
            std::cerr << "[VoronoiModelRuntime] Failed to allocate Voronoi candidate buffer" << std::endl;
            memoryAllocator.free(triIdxStagingBuffer, triIdxStagingOffset);
            cleanup();
            return false;
        }
        voronoiCandidateBuffer = candidateHandle;
        voronoiCandidateBufferOffset = candidateOffset;

        std::vector<uint32_t> candidateInitData(static_cast<size_t>(triangleCount) * K_CANDIDATES, 0xFFFFFFFFu);
        void* candidateStagingData = nullptr;
        if (createStagingBuffer(
                memoryAllocator,
                candidateBufferSize,
                candidateStagingBuffer,
                candidateStagingOffset,
                &candidateStagingData) != VK_SUCCESS ||
            !candidateStagingData) {
            std::cerr << "[VoronoiModelRuntime] Failed to create Voronoi candidate staging buffer" << std::endl;
            memoryAllocator.free(triIdxStagingBuffer, triIdxStagingOffset);
            cleanup();
            return false;
        }
        std::memcpy(candidateStagingData, candidateInitData.data(), static_cast<size_t>(candidateBufferSize));
    }

    VkCommandBuffer cmd = renderCommandPool.beginCommands();
    {
        VkBufferCopy region{};
        region.srcOffset = triIdxStagingOffset;
        region.dstOffset = triangleIndicesBufferOffset;
        region.size = triangleIndicesBufferSize;
        vkCmdCopyBuffer(cmd, triIdxStagingBuffer, triangleIndicesBuffer, 1, &region);
    }
    if (candidateStagingBuffer != VK_NULL_HANDLE) {
        VkBufferCopy region{};
        region.srcOffset = candidateStagingOffset;
        region.dstOffset = voronoiCandidateBufferOffset;
        region.size = candidateBufferSize;
        vkCmdCopyBuffer(cmd, candidateStagingBuffer, voronoiCandidateBuffer, 1, &region);
    }
    renderCommandPool.endCommands(cmd);

    memoryAllocator.free(triIdxStagingBuffer, triIdxStagingOffset);
    if (candidateStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(candidateStagingBuffer, candidateStagingOffset);
    }

    return true;
}

bool VoronoiModelRuntime::createSurfaceBuffers() {
    if (surfaceVertices.empty()) {
        std::cerr << "[VoronoiModelRuntime] Missing surface state for model" << std::endl;
        return false;
    }
    const size_t vertexCount = surfaceVertices.size();
    if (vertexCount == 0) {
        std::cerr << "[VoronoiModelRuntime] Model has 0 surface vertices" << std::endl;
        return false;
    }

    if (surfaceBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(surfaceBuffer, surfaceBufferOffset);
        surfaceBuffer = VK_NULL_HANDLE;
        surfaceBufferOffset = 0;
    }

    if (initStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(initStagingBuffer, initStagingOffset);
        initStagingBuffer = VK_NULL_HANDLE;
        initStagingOffset = 0;
        initBufferSize = 0;
    }

    const VkDeviceSize vertexBufferSize = sizeof(SurfaceVertex) * vertexCount;
    if (createStorageBuffer(
            memoryAllocator,
            vulkanDevice,
            nullptr,
            vertexBufferSize,
            surfaceBuffer,
            surfaceBufferOffset,
            nullptr,
            false,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) != VK_SUCCESS) {
        std::cerr << "[VoronoiModelRuntime] Failed to create surface vertex buffer" << std::endl;
        cleanup();
        return false;
    }

    initBufferSize = vertexBufferSize;
    void* initStagingData = nullptr;
    if (createStagingBuffer(
            memoryAllocator,
            initBufferSize,
            initStagingBuffer,
            initStagingOffset,
            &initStagingData) != VK_SUCCESS ||
        !initStagingData) {
        std::cerr << "[VoronoiModelRuntime] Failed to create surface initialization staging buffer" << std::endl;
        cleanup();
        return false;
    }
    std::memcpy(initStagingData, surfaceVertices.data(), static_cast<size_t>(initBufferSize));
    return true;
}

bool VoronoiModelRuntime::resetSurfaceState() {
    cleanupStagingBuffers();
    return createSurfaceBuffers();
}

void VoronoiModelRuntime::stageGMLSSurfaceData(
    const std::vector<voronoi::GMLSSurfaceStencil>& stencils,
    const std::vector<voronoi::GMLSSurfaceWeight>& valueWeights,
    const std::vector<voronoi::GMLSSurfaceGradientWeight>& gradientWeights) {
    if (stencils.empty() || intrinsicVertexCount == 0 || stencils.size() != intrinsicVertexCount) {
        return;
    }

    auto freeBuffer = [this](VkBuffer& buffer, VkDeviceSize& offset) {
        if (buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(buffer, offset);
            buffer = VK_NULL_HANDLE;
            offset = 0;
        }
    };
    auto freeStagingBuffer = [this](VkBuffer& buffer, VkDeviceSize& offset, VkDeviceSize& size) {
        if (buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(buffer, offset);
            buffer = VK_NULL_HANDLE;
            offset = 0;
        }
        size = 0;
    };
    auto stageBuffer = [this, &freeBuffer, &freeStagingBuffer](
                           const void* srcData,
                           VkDeviceSize bufferSize,
                           VkBuffer& deviceBuffer,
                           VkDeviceSize& deviceOffset,
                           VkBuffer& stagingBuffer,
                           VkDeviceSize& stagingOffset,
                           VkDeviceSize& stagedSize) -> bool {
        freeStagingBuffer(stagingBuffer, stagingOffset, stagedSize);
        freeBuffer(deviceBuffer, deviceOffset);

        if (bufferSize == 0 || srcData == nullptr) {
            return true;
        }

        if (createStorageBuffer(
                memoryAllocator,
                vulkanDevice,
                nullptr,
                bufferSize,
                deviceBuffer,
                deviceOffset,
                nullptr,
                false,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT) != VK_SUCCESS ||
            deviceBuffer == VK_NULL_HANDLE) {
            std::cerr << "[VoronoiModelRuntime] Failed to allocate GMLS surface device buffer" << std::endl;
            return false;
        }

        void* stagingData = nullptr;
        if (createStagingBuffer(
                memoryAllocator,
                bufferSize,
                stagingBuffer,
                stagingOffset,
                &stagingData) != VK_SUCCESS ||
            !stagingData) {
            std::cerr << "[VoronoiModelRuntime] Failed to create GMLS surface staging buffer" << std::endl;
            freeBuffer(deviceBuffer, deviceOffset);
            return false;
        }

        std::memcpy(stagingData, srcData, static_cast<size_t>(bufferSize));
        stagedSize = bufferSize;
        return true;
    };

    if (!stageBuffer(
            stencils.data(),
            sizeof(voronoi::GMLSSurfaceStencil) * stencils.size(),
            gmlsSurfaceStencilBuffer,
            gmlsSurfaceStencilBufferOffset,
            gmlsSurfaceStencilStagingBuffer,
            gmlsSurfaceStencilStagingOffset,
            gmlsSurfaceStencilBufferSize)) {
        return;
    }
    if (!stageBuffer(
            valueWeights.data(),
            sizeof(voronoi::GMLSSurfaceWeight) * valueWeights.size(),
            gmlsSurfaceWeightBuffer,
            gmlsSurfaceWeightBufferOffset,
            gmlsSurfaceWeightStagingBuffer,
            gmlsSurfaceWeightStagingOffset,
            gmlsSurfaceWeightBufferSize)) {
        return;
    }
    if (!stageBuffer(
            gradientWeights.data(),
            sizeof(voronoi::GMLSSurfaceGradientWeight) * gradientWeights.size(),
            gmlsSurfaceGradientWeightBuffer,
            gmlsSurfaceGradientWeightBufferOffset,
            gmlsSurfaceGradientWeightStagingBuffer,
            gmlsSurfaceGradientWeightStagingOffset,
            gmlsSurfaceGradientWeightBufferSize)) {
        return;
    }
}

void VoronoiModelRuntime::executeBufferTransfers(VkCommandBuffer commandBuffer) {
    if (initStagingBuffer != VK_NULL_HANDLE) {
        VkBufferCopy surfaceCopyRegion{ initStagingOffset, surfaceBufferOffset, initBufferSize };
        vkCmdCopyBuffer(commandBuffer, initStagingBuffer, surfaceBuffer, 1, &surfaceCopyRegion);
    }

    if (gmlsSurfaceStencilStagingBuffer != VK_NULL_HANDLE && gmlsSurfaceStencilBuffer != VK_NULL_HANDLE) {
        VkBufferCopy copyRegion{
            gmlsSurfaceStencilStagingOffset,
            gmlsSurfaceStencilBufferOffset,
            gmlsSurfaceStencilBufferSize
        };
        vkCmdCopyBuffer(commandBuffer, gmlsSurfaceStencilStagingBuffer, gmlsSurfaceStencilBuffer, 1, &copyRegion);
    }
    if (gmlsSurfaceWeightStagingBuffer != VK_NULL_HANDLE && gmlsSurfaceWeightBuffer != VK_NULL_HANDLE) {
        VkBufferCopy copyRegion{
            gmlsSurfaceWeightStagingOffset,
            gmlsSurfaceWeightBufferOffset,
            gmlsSurfaceWeightBufferSize
        };
        vkCmdCopyBuffer(commandBuffer, gmlsSurfaceWeightStagingBuffer, gmlsSurfaceWeightBuffer, 1, &copyRegion);
    }
    if (gmlsSurfaceGradientWeightStagingBuffer != VK_NULL_HANDLE && gmlsSurfaceGradientWeightBuffer != VK_NULL_HANDLE) {
        VkBufferCopy copyRegion{
            gmlsSurfaceGradientWeightStagingOffset,
            gmlsSurfaceGradientWeightBufferOffset,
            gmlsSurfaceGradientWeightBufferSize
        };
        vkCmdCopyBuffer(commandBuffer, gmlsSurfaceGradientWeightStagingBuffer, gmlsSurfaceGradientWeightBuffer, 1, &copyRegion);
    }
}

void VoronoiModelRuntime::cleanup() {
    auto freeBuffer = [this](VkBuffer& buffer, VkDeviceSize& offset) {
        if (buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(buffer, offset);
            buffer = VK_NULL_HANDLE;
            offset = 0;
        }
    };

    if (triangleIndicesBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(triangleIndicesBuffer, triangleIndicesBufferOffset);
        triangleIndicesBuffer = VK_NULL_HANDLE;
        triangleIndicesBufferOffset = 0;
    }
    if (voronoiCandidateBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(voronoiCandidateBuffer, voronoiCandidateBufferOffset);
        voronoiCandidateBuffer = VK_NULL_HANDLE;
        voronoiCandidateBufferOffset = 0;
    }
    if (gmlsSurfaceStencilBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceStencilBuffer, gmlsSurfaceStencilBufferOffset);
        gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
        gmlsSurfaceStencilBufferOffset = 0;
    }
    if (gmlsSurfaceWeightBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceWeightBuffer, gmlsSurfaceWeightBufferOffset);
        gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
        gmlsSurfaceWeightBufferOffset = 0;
    }
    if (gmlsSurfaceGradientWeightBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceGradientWeightBuffer, gmlsSurfaceGradientWeightBufferOffset);
        gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
        gmlsSurfaceGradientWeightBufferOffset = 0;
    }

    if (surfaceBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(surfaceBuffer, surfaceBufferOffset);
        surfaceBuffer = VK_NULL_HANDLE;
        surfaceBufferOffset = 0;
    }

    cleanupStagingBuffers();
}

void VoronoiModelRuntime::cleanupStagingBuffers() {
    if (initStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(initStagingBuffer, initStagingOffset);
        initStagingBuffer = VK_NULL_HANDLE;
        initStagingOffset = 0;
        initBufferSize = 0;
    }

    if (gmlsSurfaceStencilStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceStencilStagingBuffer, gmlsSurfaceStencilStagingOffset);
        gmlsSurfaceStencilStagingBuffer = VK_NULL_HANDLE;
        gmlsSurfaceStencilStagingOffset = 0;
        gmlsSurfaceStencilBufferSize = 0;
    }
    if (gmlsSurfaceWeightStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceWeightStagingBuffer, gmlsSurfaceWeightStagingOffset);
        gmlsSurfaceWeightStagingBuffer = VK_NULL_HANDLE;
        gmlsSurfaceWeightStagingOffset = 0;
        gmlsSurfaceWeightBufferSize = 0;
    }
    if (gmlsSurfaceGradientWeightStagingBuffer != VK_NULL_HANDLE) {
        memoryAllocator.free(gmlsSurfaceGradientWeightStagingBuffer, gmlsSurfaceGradientWeightStagingOffset);
        gmlsSurfaceGradientWeightStagingBuffer = VK_NULL_HANDLE;
        gmlsSurfaceGradientWeightStagingOffset = 0;
        gmlsSurfaceGradientWeightBufferSize = 0;
    }
}

void VoronoiModelRuntime::setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree) {
    stencilKDTree = std::move(kdTree);
}

std::vector<glm::vec3> VoronoiModelRuntime::getIntrinsicSurfacePositions() const {
    std::vector<glm::vec3> positions;
    positions.reserve(surfaceVertices.size());
    for (const SurfaceVertex& vertex : surfaceVertices) {
        positions.push_back(glm::vec3(vertex.position));
    }
    return positions;
}

VkBufferView VoronoiModelRuntime::getSupportingHalfedgeView() const {
    return supportingHalfedgeView;
}

VkBufferView VoronoiModelRuntime::getSupportingAngleView() const {
    return supportingAngleView;
}

VkBufferView VoronoiModelRuntime::getHalfedgeView() const {
    return halfedgeView;
}

VkBufferView VoronoiModelRuntime::getEdgeView() const {
    return edgeView;
}

VkBufferView VoronoiModelRuntime::getTriangleView() const {
    return triangleView;
}

VkBufferView VoronoiModelRuntime::getLengthView() const {
    return lengthView;
}

VkBufferView VoronoiModelRuntime::getInputHalfedgeView() const {
    return inputHalfedgeView;
}

VkBufferView VoronoiModelRuntime::getInputEdgeView() const {
    return inputEdgeView;
}

VkBufferView VoronoiModelRuntime::getInputTriangleView() const {
    return inputTriangleView;
}

VkBufferView VoronoiModelRuntime::getInputLengthView() const {
    return inputLengthView;
}