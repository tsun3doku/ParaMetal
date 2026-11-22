#include "VoronoiSystemBuildStage.hpp"

#include "renderers/PointRenderer.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "voronoi/VoronoiCandidate.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/VoronoiRuntime.hpp"
#include "voronoi/cuda/KNNNeighbors.hpp"
#include "voronoi/cuda/RVD.hpp"
#include "voronoi/cuda/VoronoiCellField.cuh"
#include "cuda/CudaBuffer.cuh"
#include "cuda/CudaVulkanDevice.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

VoronoiSystemBuildStage::VoronoiSystemBuildStage(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool)
    : vulkanDevice(vulkanDevice), memoryAllocator(memoryAllocator), commandPool(commandPool) {
    knn = std::make_unique<voronoi::KNNNeighbors>();
    if (!knn->initialize(vulkanDevice)) knn.reset();

    rvd = std::make_unique<voronoi::RVD>();
    if (!rvd->initialize(vulkanDevice)) rvd.reset();

    candidate = std::make_unique<voronoi::VoronoiCandidate>(vulkanDevice, memoryAllocator, commandPool);
}

VoronoiSystemBuildStage::~VoronoiSystemBuildStage() {
    cleanup();
}

bool VoronoiSystemBuildStage::buildGlobalDomain(VoronoiRuntime& runtime) {
    const auto& runtimeModelIds = runtime.getGlobalRemeshRuntimeModelIds();
    const auto& positions = runtime.getGlobalRemeshPositions();
    const auto& triangleIndices = runtime.getGlobalRemeshTriangleIndices();
    const std::vector<glm::vec4>& seeds = runtime.getSeedPositions();
    const std::array<glm::vec3, 8>& corners = runtime.getPointDomainCorners();
    if (!rvd ||
        runtimeModelIds.empty() ||
        seeds.size() < 4u ||
        positions.size() != runtimeModelIds.size() ||
        triangleIndices.size() != runtimeModelIds.size()) {
        if (runtimeModelIds.empty()) {
            std::cerr << "[VoronoiBuild] Global domain requires at least one remesh instance"
                      << std::endl;
        } else if (seeds.size() < 4u) {
            std::cerr << "[VoronoiBuild] Global domain requires at least 4 seeds" << std::endl;
        } else {
            std::cerr << "[VoronoiBuild] Global domain input size mismatch" << std::endl;
        }
        return false;
    }

    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(-std::numeric_limits<float>::max());
    for (const glm::vec3& corner : corners) {
        boundsMin = glm::min(boundsMin, corner);
        boundsMax = glm::max(boundsMax, corner);
    }
    const glm::vec3 extent = boundsMax - boundsMin;
    if (!(extent.x > 0.0f) || !(extent.y > 0.0f) || !(extent.z > 0.0f) ||
        !std::isfinite(extent.x) || !std::isfinite(extent.y) || !std::isfinite(extent.z)) {
        std::cerr << "[VoronoiBuild] Global domain bounds are degenerate" << std::endl;
        return false;
    }
    const float longestExtent = std::max(std::max(extent.x, extent.y), extent.z);
    const int resolution = runtime.getVoxelResolution();
    if (resolution < 2) return false;
    const float spacing = longestExtent / static_cast<float>(resolution - 1);
    if (!(spacing > 0.0f) || !std::isfinite(spacing)) return false;
    const float padding = runtime.getGlobalVolumePadding() + spacing;
    const glm::vec3 gridMin = boundsMin - glm::vec3(padding);
    const glm::vec3 gridSize = extent + glm::vec3(2.0f * padding);
    const glm::ivec3 gridDim(
        std::max(2, int(std::ceil(gridSize.x / spacing))),
        std::max(2, int(std::ceil(gridSize.y / spacing))),
        std::max(2, int(std::ceil(gridSize.z / spacing))));

    std::vector<voronoi::ModelGeometry> geometries;
    geometries.reserve(runtimeModelIds.size());

    for (size_t i = 0; i < runtimeModelIds.size(); ++i) {
        geometries.push_back({
            runtimeModelIds[i],
            &positions[i],
            &triangleIndices[i],
            static_cast<uint32_t>(runtime.getVoxelResolution())});
    }

    const int cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
    if (cudaDevice < 0 || cudaSetDevice(cudaDevice) != cudaSuccess) {
        std::cerr << "[VoronoiBuild] no matching CUDA device" << std::endl;
        return false;
    }

    if (!knn || !knn->buildSpatialIndex(seeds)) {
        std::cerr << "[VoronoiBuild] KNN buildSpatialIndex failed" << std::endl;
        return false;
    }

    const size_t solidMaskCount =
        size_t(gridDim.x) * size_t(gridDim.y) * size_t(gridDim.z);
    cudaUtils::CudaBuffer<uint8_t> devSolidMask;
    if (!devSolidMask.allocate(solidMaskCount)) {
        std::cerr << "[VoronoiBuild] failed to allocate solid mask" << std::endl;
        return false;
    }

    voronoi::RVDConfig config{};
    config.contactRange = runtime.getContactRange();
    config.lloydIterations = static_cast<uint32_t>(
        std::max(0, runtime.getLloydIterations()));
    config.lloydVolumeThresholdFraction = runtime.getLloydVolumeThreshold();
    config.lloydDamping = runtime.getLloydDamping();

    voronoi::RVDInput input{};
    input.seeds = &seeds;
    input.seedSpatialIndex = knn.get();
    input.domainCorners = &corners;
    input.geometries = &geometries;
    input.outDevSolidMask = devSolidMask.get();
    input.solidMaskGridMin = gridMin;
    input.solidMaskGridDim = gridDim;
    input.solidMaskCellSize = spacing;

    voronoi::RVDResult result;
    if (!rvd->build(input, config, result)) {
        std::cerr << "[VoronoiBuild] RVD build failed" << std::endl;
        return false;
    }

    if (result.seedPositions.empty()) {
        result.seedPositions = seeds;
    }
    const std::vector<glm::vec4> resolvedSeeds = result.seedPositions;
    if (!runtime.materializeTopology(
            std::move(result),
            runtimeModelIds,
            boundsMin,
            boundsMax,
            gridMin,
            gridDim,
            spacing)) {
        std::cerr << "[VoronoiBuild] Runtime topology materialization failed" << std::endl;
        return false;
    }

    outGlobalSeedIdGridCuda.cleanup();
    outGlobalSeedIdGridExternal.cleanup();
    outGlobalSeedIdGridBuffer = VK_NULL_HANDLE;
    outGlobalSeedIdGridBufferOffset = 0;

    {
        const size_t totalVoxels =
            size_t(gridDim.x) * size_t(gridDim.y) * size_t(gridDim.z);
        if (totalVoxels > 0 && cudaDevice >= 0 &&
            cudaSetDevice(cudaDevice) == cudaSuccess &&
            outGlobalSeedIdGridExternal.initialize(
                vulkanDevice,
                VkDeviceSize(totalVoxels) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) &&
            outGlobalSeedIdGridCuda.ensureMapped(outGlobalSeedIdGridExternal, cudaDevice)) {
            uint32_t* devSeedIds =
                reinterpret_cast<uint32_t*>(outGlobalSeedIdGridCuda.getPointer());
            if (voronoi::buildSeedIdGrid(
                    knn->deviceKnn(),
                    devSolidMask.get(),
                    resolvedSeeds,
                    gridMin,
                    gridDim,
                    spacing,
                    devSeedIds,
                    0)) {
                outGlobalSeedIdGridBuffer = outGlobalSeedIdGridExternal.getBuffer();
                outGlobalSeedIdGridBufferOffset = 0;
            } else {
                std::cerr << "[VoronoiBuild] warning: buildSeedIdGrid failed; fluid volume visualization unavailable" << std::endl;
            }
        } else {
            std::cerr << "[VoronoiBuild] warning: GPU seed ID grid allocation failed" << std::endl;
        }
    }

    runtime.setSeedPositions(resolvedSeeds, corners);

    return true;
}

bool VoronoiSystemBuildStage::buildGlobalDisplayCandidates(VoronoiRuntime& runtime) {
    const auto& seeds = runtime.getSeedPositions();
    const auto& runtimeModelIds = runtime.getGlobalRemeshRuntimeModelIds();
    const auto& surfacePositions = runtime.getGlobalRemeshSurfacePositions();
    const auto& surfaceTriangleIndices = runtime.getGlobalRemeshSurfaceTriangleIndices();
    if (!knn ||
        seeds.size() < 2u ||
        runtimeModelIds.empty() ||
        surfacePositions.size() != runtimeModelIds.size() ||
        surfaceTriangleIndices.size() != runtimeModelIds.size()) {
        return false;
    }

    outCandidateNodeCount = static_cast<uint32_t>(seeds.size());
    const uint32_t neighborCount = std::min(uint32_t{50}, outCandidateNodeCount - 1u);
    if (neighborCount == 0) return false;

    std::vector<uint32_t> knnRows;
    knnRows.reserve(outCandidateNodeCount);
    for (uint32_t cell = 0; cell < outCandidateNodeCount; ++cell) {
        knnRows.push_back(cell);
    }
    if (!knn->buildNeighbors(neighborCount, knnRows)) return false;

    std::vector<uint32_t> neighborIndices;
    if (!knn->downloadPrefix(neighborCount, neighborIndices)) return false;

    if (!uploadCandidateInputs(seeds, neighborIndices)) return false;

    if (!candidate || !candidate->build(runtimeModelIds, surfacePositions, surfaceTriangleIndices)) {
        return false;
    }
    return candidate->getModelCount() > 0;
}

const std::vector<uint32_t>& VoronoiSystemBuildStage::getGlobalDisplayFaceCounts() const {
    static const std::vector<uint32_t> empty;
    return candidate ? candidate->getFaceCounts() : empty;
}

const std::vector<VkBuffer>& VoronoiSystemBuildStage::getGlobalDisplayVertexBuffers() const {
    static const std::vector<VkBuffer> empty;
    return candidate ? candidate->getVertexBuffers() : empty;
}

const std::vector<VkDeviceSize>& VoronoiSystemBuildStage::getGlobalDisplayVertexBufferOffsets() const {
    static const std::vector<VkDeviceSize> empty;
    return candidate ? candidate->getVertexBufferOffsets() : empty;
}

const std::vector<VkBuffer>& VoronoiSystemBuildStage::getGlobalDisplayFaceIndexBuffers() const {
    static const std::vector<VkBuffer> empty;
    return candidate ? candidate->getFaceIndexBuffers() : empty;
}

const std::vector<VkDeviceSize>& VoronoiSystemBuildStage::getGlobalDisplayFaceIndexBufferOffsets() const {
    static const std::vector<VkDeviceSize> empty;
    return candidate ? candidate->getFaceIndexBufferOffsets() : empty;
}

const std::vector<VkBuffer>& VoronoiSystemBuildStage::getGlobalDisplayCandidateBuffers() const {
    static const std::vector<VkBuffer> empty;
    return candidate ? candidate->getCandidateBuffers() : empty;
}

const std::vector<VkDeviceSize>& VoronoiSystemBuildStage::getGlobalDisplayCandidateBufferOffsets() const {
    static const std::vector<VkDeviceSize> empty;
    return candidate ? candidate->getCandidateBufferOffsets() : empty;
}

bool VoronoiSystemBuildStage::uploadCandidateInputs(const std::vector<glm::vec4>& positions, const std::vector<uint32_t>& neighborIndices) {
    if (positions.empty() || neighborIndices.empty()) return false;
    const VkDeviceSize alignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    VkBuffer newNeighborBuffer = VK_NULL_HANDLE;
    VkDeviceSize newNeighborOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, neighborIndices.data(),
                           neighborIndices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           alignment, newNeighborBuffer, newNeighborOffset) != VK_SUCCESS) return false;

    VkBuffer newSeedBuffer = VK_NULL_HANDLE;
    VkDeviceSize newSeedOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, positions.data(),
                           positions.size() * sizeof(glm::vec4),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           alignment, newSeedBuffer, newSeedOffset) != VK_SUCCESS) {
        memoryAllocator.free(newNeighborBuffer, newNeighborOffset);
        return false;
    }

    std::vector<PointRenderer::PointVertex> pointVertices;
    pointVertices.reserve(positions.size());
    for (const glm::vec4& p : positions) {
        pointVertices.push_back({
            glm::vec3(p),
            glm::vec3(1.0f, 0.0f, 0.0f)
        });
    }

    VkBuffer newPointVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize newPointVertexOffset = 0;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, pointVertices.data(),
                           pointVertices.size() * sizeof(PointRenderer::PointVertex),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           alignof(PointRenderer::PointVertex),
                           newPointVertexBuffer, newPointVertexOffset) != VK_SUCCESS) {
        memoryAllocator.free(newNeighborBuffer, newNeighborOffset);
        memoryAllocator.free(newSeedBuffer, newSeedOffset);
        return false;
    }

    if (outCandidateNeighborIndicesBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outCandidateNeighborIndicesBuffer, outCandidateNeighborIndicesBufferOffset);
    }
    if (outSeedPositionBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outSeedPositionBuffer, outSeedPositionBufferOffset);
    }
    if (outCandidatePointVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outCandidatePointVertexBuffer, outCandidatePointVertexBufferOffset);
    }

    outCandidateNeighborIndicesBuffer = newNeighborBuffer;
    outCandidateNeighborIndicesBufferOffset = newNeighborOffset;
    outSeedPositionBuffer = newSeedBuffer;
    outSeedPositionBufferOffset = newSeedOffset;
    outCandidatePointVertexBuffer = newPointVertexBuffer;
    outCandidatePointVertexBufferOffset = newPointVertexOffset;
    return true;
}

void VoronoiSystemBuildStage::copyGpu(VoronoiProduct& product) const {
    product.candidateNodeCount = outCandidateNodeCount;
    product.candidateNeighborIndicesBuffer = outCandidateNeighborIndicesBuffer;
    product.candidateNeighborIndicesBufferOffset = outCandidateNeighborIndicesBufferOffset;
    product.seedPositionBuffer = outSeedPositionBuffer;
    product.seedPositionBufferOffset = outSeedPositionBufferOffset;
    product.candidatePointVertexBuffer = outCandidatePointVertexBuffer;
    product.candidatePointVertexBufferOffset = outCandidatePointVertexBufferOffset;
    product.globalSeedIdGridBuffer = outGlobalSeedIdGridBuffer;
    product.globalSeedIdGridBufferOffset = outGlobalSeedIdGridBufferOffset;

    if (candidate) {
        candidate->copyGpu(product);
    }
}

void VoronoiSystemBuildStage::cleanup() {
    if (knn) knn->cleanup();
    if (rvd) rvd->cleanup();

    if (outCandidateNeighborIndicesBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outCandidateNeighborIndicesBuffer, outCandidateNeighborIndicesBufferOffset);
        outCandidateNeighborIndicesBuffer = VK_NULL_HANDLE;
        outCandidateNeighborIndicesBufferOffset = 0;
    }
    if (outSeedPositionBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outSeedPositionBuffer, outSeedPositionBufferOffset);
        outSeedPositionBuffer = VK_NULL_HANDLE;
        outSeedPositionBufferOffset = 0;
    }
    if (outCandidatePointVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outCandidatePointVertexBuffer, outCandidatePointVertexBufferOffset);
        outCandidatePointVertexBuffer = VK_NULL_HANDLE;
        outCandidatePointVertexBufferOffset = 0;
    }
    outGlobalSeedIdGridCuda.cleanup();
    outGlobalSeedIdGridExternal.cleanup();
    outGlobalSeedIdGridBuffer = VK_NULL_HANDLE;
    outGlobalSeedIdGridBufferOffset = 0;

    if (candidate) {
        candidate->cleanup();
    }

    outCandidateNodeCount = 0;
}



