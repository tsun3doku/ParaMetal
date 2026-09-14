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
#include "voronoi/cuda/GlobalSdf.cuh"
#include "voronoi/cuda/VoronoiCellField.cuh"
#include "cuda/CudaBuffer.cuh"

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
    const float padding = runtime.getGlobalSdfPadding() + spacing;
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
    const size_t sampleCount64 = size_t(gridDim.x) * gridDim.y * gridDim.z;
    cudaUtils::CudaBuffer<float> devUnionSdf;
    if (!devUnionSdf.allocate(sampleCount64)) {
        std::cerr << "[VoronoiBuild] failed to allocate devUnionSdf" << std::endl;
        return false;
    }

    voronoi::globalsdf::BuildInput sdfInput{};
    sdfInput.geometries = &geometries;
    sdfInput.gridMin = gridMin;
    sdfInput.gridDim = gridDim;
    sdfInput.cellSize = spacing;
    sdfInput.outDevUnionSdf = devUnionSdf.get();
    voronoi::globalsdf::BuildResult sdfResult;
    if (!voronoi::globalsdf::build(sdfInput, sdfResult, 0)) {
        std::cerr << "[VoronoiBuild] GlobalSdf build failed" << std::endl;
        return false;
    }

    if (!knn || !knn->buildSpatialIndex(seeds)) {
        std::cerr << "[VoronoiBuild] KNN buildSpatialIndex failed" << std::endl;
        return false;
    }

    if (globalSeedIdGridBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, globalSeedIdGridBuffer, globalSeedIdGridBufferOffset);
        globalSeedIdGridBuffer = VK_NULL_HANDLE;
        globalSeedIdGridBufferOffset = 0;
        globalSeedIdGridCount = 0;
    }

    std::vector<uint32_t> seedIds;
    if (voronoi::buildSeedIdGrid(knn->deviceKnn(), devUnionSdf.get(), seeds, gridMin, gridDim, spacing, seedIds, 0)) {
        VkBuffer newSeedGridBuffer = VK_NULL_HANDLE;
        VkDeviceSize newSeedGridOffset = 0;
        if (uploadDeviceBuffer(memoryAllocator, commandPool, seedIds.data(),
                               seedIds.size() * sizeof(uint32_t),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               sizeof(uint32_t),
                               newSeedGridBuffer, newSeedGridOffset) == VK_SUCCESS) {
            globalSeedIdGridBuffer = newSeedGridBuffer;
            globalSeedIdGridBufferOffset = newSeedGridOffset;
            globalSeedIdGridCount = static_cast<uint32_t>(seedIds.size());
        } else {
            std::cerr << "[VoronoiBuild] failed to upload seed ID grid buffer" << std::endl;
        }
    } else {
        std::cerr << "[VoronoiBuild] warning: buildSeedIdGrid failed or disabled; fluid volume visualization unavailable" << std::endl;
    }

    devUnionSdf.reset();

    voronoi::RVDConfig config{};
    config.restrictedChunkSize = 512;
    config.contactRange = runtime.getContactRange();

    voronoi::RVDInput input{};
    input.seeds = &seeds;
    input.seedSpatialIndex = knn.get();
    input.domainCorners = &corners;
    input.geometries = &geometries;
    input.nominalCellSize = runtime.getCellSize();

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
        globalFragmentVolumes[i] = static_cast<float>(result.fragments[i].volume);
    }

    globalNominalCellSize = input.nominalCellSize;

    globalFaceFragmentA.clear();
    globalFaceFragmentB.clear();
    globalFaceAreas.clear();
    globalFaceOrthogonalDistancesA.clear();
    globalFaceOrthogonalDistancesB.clear();
    globalFaceNormals.clear();

    globalBoundaryFaceFragments.clear();
    globalBoundaryFaceDomainFaces.clear();
    globalBoundaryFaceAreas.clear();
    globalBoundaryFaceOrthogonalDistances.clear();

    for (const auto& f : result.fragmentFaces) {
        if (f.source == voronoi::FragmentFaceSource::DomainBoundary) {
            globalBoundaryFaceFragments.push_back(f.fragmentA);
            globalBoundaryFaceDomainFaces.push_back(f.domainFace);
            globalBoundaryFaceAreas.push_back(static_cast<float>(f.area));
            globalBoundaryFaceOrthogonalDistances.push_back(static_cast<float>(f.orthogonalDistanceA));
        } else {
            uint32_t a = f.fragmentA;
            uint32_t b = f.fragmentB;
            float dA = static_cast<float>(f.orthogonalDistanceA);
            float dB = static_cast<float>(f.orthogonalDistanceB);
            glm::vec3 normal = f.normal;
            if (b < a) {
                std::swap(a, b);
                std::swap(dA, dB);
                normal = -normal;
            }
            globalFaceFragmentA.push_back(a);
            globalFaceFragmentB.push_back(b);
            globalFaceAreas.push_back(static_cast<float>(f.area));
            globalFaceOrthogonalDistancesA.push_back(dA);
            globalFaceOrthogonalDistancesB.push_back(dB);
            globalFaceNormals.push_back(normal);
        }
    }

    const size_t contactFaceCount = result.contactFaces.size();
    globalContactFaceFragmentA.resize(contactFaceCount);
    globalContactFaceFragmentB.resize(contactFaceCount);
    globalContactFaceAreas.resize(contactFaceCount);
    globalContactFaceOrthogonalDistancesA.resize(contactFaceCount);
    globalContactFaceOrthogonalDistancesB.resize(contactFaceCount);
    globalContactFaceGaps.resize(contactFaceCount);
    for (size_t i = 0; i < contactFaceCount; ++i) {
        uint32_t a = result.contactFaces[i].fragmentA;
        uint32_t b = result.contactFaces[i].fragmentB;
        float dA = static_cast<float>(result.contactFaces[i].distanceA);
        float dB = static_cast<float>(result.contactFaces[i].distanceB);
        if (b < a) {
            std::swap(a, b);
            std::swap(dA, dB);
        }
        globalContactFaceFragmentA[i] = a;
        globalContactFaceFragmentB[i] = b;
        globalContactFaceAreas[i] = static_cast<float>(result.contactFaces[i].area);
        globalContactFaceOrthogonalDistancesA[i] = dA;
        globalContactFaceOrthogonalDistancesB[i] = dB;
        globalContactFaceGaps[i] = static_cast<float>(result.contactFaces[i].gap);
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

    globalSdfValues = sdfResult.values;
    globalSdfRuntimeModelIds = runtimeModelIds;
    globalSdfGridMin = gridMin;
    globalSdfGridDim = gridDim;
    globalSdfCellSize = spacing;
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

void VoronoiSystemBuildStage::releaseGpuResources(VoronoiProduct& product) {
    product.candidateNodeCount = candidateNodeCount;
    candidateNodeCount = 0;

    product.candidateNeighborIndicesBuffer = candidateNeighborIndicesBuffer;
    product.candidateNeighborIndicesBufferOffset = candidateNeighborIndicesBufferOffset;
    candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    candidateNeighborIndicesBufferOffset = 0;

    product.seedPositionBuffer = seedPositionBuffer;
    product.seedPositionBufferOffset = seedPositionBufferOffset;
    seedPositionBuffer = VK_NULL_HANDLE;
    seedPositionBufferOffset = 0;

    product.candidatePointVertexBuffer = candidatePointVertexBuffer;
    product.candidatePointVertexBufferOffset = candidatePointVertexBufferOffset;
    candidatePointVertexBuffer = VK_NULL_HANDLE;
    candidatePointVertexBufferOffset = 0;

    product.contactVertexBuffer = contactVertexBuffer;
    product.contactVertexBufferOffset = contactVertexBufferOffset;
    product.contactVertexCount = contactVertexCount;
    contactVertexBuffer = VK_NULL_HANDLE;
    contactVertexBufferOffset = 0;
    contactVertexCount = 0;

    product.globalSeedIdGridBuffer = globalSeedIdGridBuffer;
    product.globalSeedIdGridBufferOffset = globalSeedIdGridBufferOffset;
    product.globalSeedIdGridCount = globalSeedIdGridCount;
    globalSeedIdGridBuffer = VK_NULL_HANDLE;
    globalSeedIdGridBufferOffset = 0;
    globalSeedIdGridCount = 0;

    if (candidate) {
        candidate->releaseGpuResources(product);
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
    if (globalSeedIdGridBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, globalSeedIdGridBuffer, globalSeedIdGridBufferOffset);
        globalSeedIdGridBuffer = VK_NULL_HANDLE;
        globalSeedIdGridBufferOffset = 0;
        globalSeedIdGridCount = 0;
    }

    if (candidate) {
        candidate->cleanup();
    }

    candidateNodeCount = 0;
    globalFragmentInstanceIds.clear();
    globalFragmentSeedIds.clear();
    globalFragmentCentroids.clear();
    globalFragmentVolumes.clear();
    globalNominalCellSize = 0.0f;
    globalFaceFragmentA.clear();
    globalFaceFragmentB.clear();
    globalFaceAreas.clear();
    globalFaceOrthogonalDistancesA.clear();
    globalFaceOrthogonalDistancesB.clear();
    globalFaceNormals.clear();
    globalContactFaceFragmentA.clear();
    globalContactFaceFragmentB.clear();
    globalContactFaceAreas.clear();
    globalContactFaceOrthogonalDistancesA.clear();
    globalContactFaceOrthogonalDistancesB.clear();
    globalContactFaceGaps.clear();
    globalBoundaryFaceFragments.clear();
    globalBoundaryFaceDomainFaces.clear();
    globalBoundaryFaceAreas.clear();
    globalBoundaryFaceOrthogonalDistances.clear();
    globalInstanceFragmentCounts.clear();
    globalSdfValues.clear();
    globalSdfRuntimeModelIds.clear();
    globalSdfGridMin = {};
    globalSdfGridDim = {};
    globalSdfCellSize = 0.0f;
    globalSeedPositions.clear();
    globalDomainCorners = {};
}