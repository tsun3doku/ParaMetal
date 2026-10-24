#include "VoronoiSystemBuildStage.hpp"

#include "renderers/PointRenderer.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "spatial/VoxelGrid.hpp"
#include "spatial/TriangleHashGrid.hpp"
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
#include <unordered_map>
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
    const float longestExtent = std::max(std::max(extent.x, extent.y), extent.z);
    if (!(longestExtent > 0.0f) || !std::isfinite(longestExtent)) return false;
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
    const glm::vec3 gridMax = gridMin + glm::vec3(gridDim) * spacing;

    std::vector<VoxelGrid> instanceGrids;
    std::vector<std::vector<std::array<glm::vec4, 3>>> instanceTriangles;
    instanceGrids.reserve(runtimeModelIds.size());
    instanceTriangles.reserve(runtimeModelIds.size());
    std::vector<voronoi::ModelGeometry> geometries;
    geometries.reserve(runtimeModelIds.size());

    for (size_t i = 0; i < runtimeModelIds.size(); ++i) {
        TriangleHashGrid triangleGrid;
        triangleGrid.build(positions[i], triangleIndices[i], gridMin, gridMax, spacing);
        VoxelGrid instanceGrid;
        instanceGrid.build(positions[i], triangleIndices[i], triangleGrid, resolution);
        std::vector<std::array<glm::vec4, 3>> triangles;
        extractMeshTriangles(positions[i], triangleIndices[i], triangles);
        instanceGrids.push_back(std::move(instanceGrid));
        instanceTriangles.push_back(std::move(triangles));
        geometries.push_back({runtimeModelIds[i], &instanceGrids.back(), &instanceTriangles.back()});
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
    config.restrictedChunkSize = 8192;
    config.contactRange = runtime.getContactRange();

    voronoi::RVDInput input{};
    input.seeds = &seeds;
    input.seedSpatialIndex = knn.get();
    input.domainCorners = &corners;
    input.geometries = &geometries;
    input.nominalCellSize = runtime.getCellSize();
    input.outDevSolidMask = devSolidMask.get();
    input.solidMaskGridMin = gridMin;
    input.solidMaskGridDim = gridDim;
    input.solidMaskCellSize = spacing;

    voronoi::RVDResult result;
    if (!rvd->build(input, config, result)) {
        std::cerr << "[VoronoiBuild] RVD build failed" << std::endl;
        return false;
    }

    if (contactVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, contactVertexBuffer, contactVertexBufferOffset);
        contactVertexBuffer = VK_NULL_HANDLE;
        contactVertexBufferOffset = 0;
        contactVertexCount = 0;
    }

    if (!result.contactMesh.empty()) {
        VkBuffer newContactBuffer = VK_NULL_HANDLE;
        VkDeviceSize newContactOffset = 0;
        if (uploadDeviceBuffer(memoryAllocator, commandPool, result.contactMesh.data(),
                               result.contactMesh.size() * sizeof(voronoi::ContactMeshVertex),
                               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               alignof(voronoi::ContactMeshVertex),
                               newContactBuffer, newContactOffset) == VK_SUCCESS) {
            contactVertexBuffer = newContactBuffer;
            contactVertexBufferOffset = newContactOffset;
            contactVertexCount = static_cast<uint32_t>(result.contactMesh.size());
        } else {
            std::cerr << "[VoronoiBuild] failed to upload contact mesh vertex buffer" << std::endl;
        }
    }

    const size_t fragmentCount = result.fragments.size();
    globalFragmentInstanceIds.resize(fragmentCount);
    globalFragmentSeedIds.resize(fragmentCount);
    globalFragmentCentroids.resize(fragmentCount);
    globalFragmentVolumes.resize(fragmentCount);
    for (size_t i = 0; i < fragmentCount; ++i) {
        globalFragmentInstanceIds[i] = result.fragments[i].ownerModelId;
        globalFragmentSeedIds[i] = result.fragments[i].cellId;
        globalFragmentCentroids[i] = result.fragments[i].centroid;
        globalFragmentVolumes[i] = result.fragments[i].volume;
    }
    globalSeedToFragment = result.seedToFragment;

    globalSeedIdGridCuda.cleanup();
    globalSeedIdGridExternal.cleanup();
    globalSeedIdGridBuffer = VK_NULL_HANDLE;
    globalSeedIdGridBufferOffset = 0;

    {
        const size_t totalVoxels =
            size_t(gridDim.x) * size_t(gridDim.y) * size_t(gridDim.z);
        if (totalVoxels > 0 && cudaDevice >= 0 &&
            cudaSetDevice(cudaDevice) == cudaSuccess &&
            globalSeedIdGridExternal.initialize(
                vulkanDevice,
                VkDeviceSize(totalVoxels) * sizeof(uint32_t),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) &&
            globalSeedIdGridCuda.ensureMapped(globalSeedIdGridExternal, cudaDevice)) {
            uint32_t* devSeedIds =
                reinterpret_cast<uint32_t*>(globalSeedIdGridCuda.getPointer());
            if (voronoi::buildSeedIdGrid(
                    knn->deviceKnn(),
                    devSolidMask.get(),
                    seeds,
                    gridMin,
                    gridDim,
                    spacing,
                    devSeedIds,
                    0)) {
                globalSeedIdGridBuffer = globalSeedIdGridExternal.getBuffer();
                globalSeedIdGridBufferOffset = 0;
            } else {
                std::cerr << "[VoronoiBuild] warning: buildSeedIdGrid failed; fluid volume visualization unavailable" << std::endl;
            }
        } else {
            std::cerr << "[VoronoiBuild] warning: GPU seed ID grid allocation failed" << std::endl;
        }
    }

    globalNominalCellSize = input.nominalCellSize;

    const size_t totalFaces = result.fragmentFaces.size();
    globalFaceFragmentA.resize(totalFaces);
    globalFaceFragmentB.resize(totalFaces);
    globalFaceAreasA.resize(totalFaces);
    globalFaceAreaVectorsA.resize(totalFaces);
    globalFaceMomentsA.resize(totalFaces);
    globalFaceAreasB.resize(totalFaces);
    globalFaceAreaVectorsB.resize(totalFaces);
    globalFaceMomentsB.resize(totalFaces);
    globalFaceTypes.resize(totalFaces);
    globalFaceDomainFaces.resize(totalFaces);

    globalBoundaryFaceIndices.clear();
    globalBoundaryFaceFragments.clear();
    globalBoundaryFaceDomainFaces.clear();

    for (size_t i = 0; i < totalFaces; ++i) {
        const auto& f = result.fragmentFaces[i];
        globalFaceFragmentA[i] = f.fragmentA;
        globalFaceFragmentB[i] = f.fragmentB;
        globalFaceAreasA[i] = f.areaA;
        globalFaceAreaVectorsA[i] = f.areaVectorA;
        globalFaceMomentsA[i] = f.momentA;
        globalFaceAreasB[i] = f.areaB;
        globalFaceAreaVectorsB[i] = f.areaVectorB;
        globalFaceMomentsB[i] = f.momentB;
        globalFaceTypes[i] = static_cast<uint8_t>(f.type);
        globalFaceDomainFaces[i] = f.domainFace;

        if (f.type == voronoi::FaceType::DomainBoundary) {
            globalBoundaryFaceIndices.push_back(static_cast<uint32_t>(i));
            globalBoundaryFaceFragments.push_back(f.fragmentA);
            globalBoundaryFaceDomainFaces.push_back(f.domainFace);
        }
    }

    const size_t couplingCount = result.contactCouplings.size();
    globalContactCouplingFaceA.resize(couplingCount);
    globalContactCouplingFaceB.resize(couplingCount);
    globalContactCouplingAreas.resize(couplingCount);
    for (size_t i = 0; i < couplingCount; ++i) {
        globalContactCouplingFaceA[i] = result.contactCouplings[i].faceA;
        globalContactCouplingFaceB[i] = result.contactCouplings[i].faceB;
        globalContactCouplingAreas[i] = result.contactCouplings[i].couplingArea;
    }

    std::unordered_map<uint32_t, uint32_t> runtimeModelIdToInstanceIndex;
    for (size_t i = 0; i < runtimeModelIds.size(); ++i) {
        runtimeModelIdToInstanceIndex[runtimeModelIds[i]] = static_cast<uint32_t>(i);
    }
    globalInstanceFragmentCounts.assign(runtimeModelIds.size(), 0);
    for (const auto& frag : result.fragments) {
        if (frag.ownerModelId != voronoi::InvalidModelId) {
            const auto it = runtimeModelIdToInstanceIndex.find(frag.ownerModelId);
            if (it != runtimeModelIdToInstanceIndex.end()) {
                globalInstanceFragmentCounts[it->second]++;
            }
        }
    }

    globalVolumeGridMin = gridMin;
    globalVolumeGridDim = gridDim;
    globalVolumeGridCellSize = spacing;
    globalSeedPositions = seeds;
    globalDomainCorners = corners;

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

    candidateNodeCount = static_cast<uint32_t>(seeds.size());
    const uint32_t neighborCount = std::min(uint32_t{50}, candidateNodeCount - 1u);
    if (neighborCount == 0) return false;

    std::vector<uint32_t> knnRows;
    knnRows.reserve(candidateNodeCount);
    for (uint32_t cell = 0; cell < candidateNodeCount; ++cell) {
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

const std::vector<uint32_t>& VoronoiSystemBuildStage::getGlobalDisplayRuntimeModelIds() const {
    static const std::vector<uint32_t> empty;
    return candidate ? candidate->getRuntimeModelIds() : empty;
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

    if (candidateNeighborIndicesBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, candidateNeighborIndicesBuffer, candidateNeighborIndicesBufferOffset);
    }
    if (seedPositionBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedPositionBuffer, seedPositionBufferOffset);
    }
    if (candidatePointVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, candidatePointVertexBuffer, candidatePointVertexBufferOffset);
    }

    candidateNeighborIndicesBuffer = newNeighborBuffer;
    candidateNeighborIndicesBufferOffset = newNeighborOffset;
    seedPositionBuffer = newSeedBuffer;
    seedPositionBufferOffset = newSeedOffset;
    candidatePointVertexBuffer = newPointVertexBuffer;
    candidatePointVertexBufferOffset = newPointVertexOffset;
    return true;
}

void VoronoiSystemBuildStage::extractMeshTriangles(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    std::vector<std::array<glm::vec4, 3>>& outMeshTriangles) const {
    outMeshTriangles.clear();
    outMeshTriangles.reserve(indices.size() / 3);
    for (size_t index = 0; index + 2 < indices.size(); index += 3) {
        const uint32_t first = indices[index];
        const uint32_t second = indices[index + 1];
        const uint32_t third = indices[index + 2];
        outMeshTriangles.push_back({glm::vec4(positions[first], 0.0f),
                                    glm::vec4(positions[second], 0.0f),
                                    glm::vec4(positions[third], 0.0f)});
    }
}

void VoronoiSystemBuildStage::copyGpu(VoronoiProduct& product) const {
    product.candidateNodeCount = candidateNodeCount;
    product.candidateNeighborIndicesBuffer = candidateNeighborIndicesBuffer;
    product.candidateNeighborIndicesBufferOffset = candidateNeighborIndicesBufferOffset;
    product.seedPositionBuffer = seedPositionBuffer;
    product.seedPositionBufferOffset = seedPositionBufferOffset;
    product.candidatePointVertexBuffer = candidatePointVertexBuffer;
    product.candidatePointVertexBufferOffset = candidatePointVertexBufferOffset;
    product.contactVertexBuffer = contactVertexBuffer;
    product.contactVertexBufferOffset = contactVertexBufferOffset;
    product.contactVertexCount = contactVertexCount;
    product.globalSeedIdGridBuffer = globalSeedIdGridBuffer;
    product.globalSeedIdGridBufferOffset = globalSeedIdGridBufferOffset;

    if (candidate) {
        candidate->copyGpu(product);
    }
}

void VoronoiSystemBuildStage::cleanup() {
    if (knn) knn->cleanup();
    if (rvd) rvd->cleanup();

    if (candidateNeighborIndicesBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, candidateNeighborIndicesBuffer, candidateNeighborIndicesBufferOffset);
        candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
        candidateNeighborIndicesBufferOffset = 0;
    }
    if (seedPositionBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedPositionBuffer, seedPositionBufferOffset);
        seedPositionBuffer = VK_NULL_HANDLE;
        seedPositionBufferOffset = 0;
    }
    if (candidatePointVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, candidatePointVertexBuffer, candidatePointVertexBufferOffset);
        candidatePointVertexBuffer = VK_NULL_HANDLE;
        candidatePointVertexBufferOffset = 0;
    }
    if (contactVertexBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, contactVertexBuffer, contactVertexBufferOffset);
        contactVertexBuffer = VK_NULL_HANDLE;
        contactVertexBufferOffset = 0;
        contactVertexCount = 0;
    }
    globalSeedIdGridCuda.cleanup();
    globalSeedIdGridExternal.cleanup();
    globalSeedIdGridBuffer = VK_NULL_HANDLE;
    globalSeedIdGridBufferOffset = 0;

    if (candidate) {
        candidate->cleanup();
    }

    candidateNodeCount = 0;
    globalFragmentInstanceIds.clear();
    globalFragmentSeedIds.clear();
    globalSeedToFragment.clear();
    globalFragmentCentroids.clear();
    globalFragmentVolumes.clear();
    globalNominalCellSize = 0.0f;
    globalFaceFragmentA.clear();
    globalFaceFragmentB.clear();
    globalFaceAreasA.clear();
    globalFaceAreaVectorsA.clear();
    globalFaceMomentsA.clear();
    globalFaceAreasB.clear();
    globalFaceAreaVectorsB.clear();
    globalFaceMomentsB.clear();
    globalFaceTypes.clear();
    globalFaceDomainFaces.clear();
    globalContactCouplingFaceA.clear();
    globalContactCouplingFaceB.clear();
    globalContactCouplingAreas.clear();
    globalBoundaryFaceIndices.clear();
    globalBoundaryFaceFragments.clear();
    globalBoundaryFaceDomainFaces.clear();
    globalInstanceFragmentCounts.clear();
    globalVolumeGridMin = {};
    globalVolumeGridDim = {};
    globalVolumeGridCellSize = 0.0f;
    globalSeedPositions.clear();
    globalDomainCorners = {};
}
