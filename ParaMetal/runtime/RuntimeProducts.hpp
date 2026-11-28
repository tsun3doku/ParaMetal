#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include "hash/HashValues.hpp"


// Borrowed view, owned by ModelRegistry
struct ModelProduct {
    uint32_t runtimeModelId = 0;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize vertexBufferOffset = 0;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexBufferOffset = 0;
    uint32_t indexCount = 0;
    VkBuffer renderVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize renderVertexBufferOffset = 0;
    VkBuffer renderIndexBuffer = VK_NULL_HANDLE;
    VkDeviceSize renderIndexBufferOffset = 0;
    uint32_t renderIndexCount = 0;
    HashValues hashes{};

    bool isValid() const {
        return runtimeModelId != 0 &&
            vertexBuffer != VK_NULL_HANDLE &&
            indexBuffer != VK_NULL_HANDLE &&
            indexCount != 0 &&
            renderVertexBuffer != VK_NULL_HANDLE &&
            renderIndexBuffer != VK_NULL_HANDLE &&
            renderIndexCount != 0;
    }

};

// Runtime view owned by RemeshSystem. GPU buffers are borrowed; CPU geometry
// arrays are generation-owned output and remain valid while this product is published.
struct RemeshProduct {
    uint32_t runtimeModelId = 0;
    std::vector<glm::vec3> geometryPositions;
    std::vector<uint32_t> geometryTriangleIndices;
    std::vector<glm::vec3> surfacePositions;
    std::vector<glm::vec3> surfaceNormals;
    std::vector<uint32_t> surfaceTriangleIndices;
    std::vector<uint32_t> surfaceFaceIds;
    std::vector<uint32_t> surfaceVertexIds;
    std::vector<uint32_t> surfaceEdgeIds;
    VkBuffer intrinsicTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicTriangleBufferOffset = 0;
    VkBuffer intrinsicVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicVertexBufferOffset = 0;
    size_t intrinsicTriangleCount = 0;
    size_t intrinsicVertexCount = 0;
    float averageTriangleArea = 0.0f;

    VkBuffer faceIdBuffer = VK_NULL_HANDLE;
    VkDeviceSize faceIdOffset = 0;
    VkBufferView faceIdView = VK_NULL_HANDLE;

    VkBuffer supportingHalfedgeBuffer = VK_NULL_HANDLE;
    VkDeviceSize supportingHalfedgeOffset = 0;
    VkBufferView supportingHalfedgeView = VK_NULL_HANDLE;

    VkBuffer supportingAngleBuffer = VK_NULL_HANDLE;
    VkDeviceSize supportingAngleOffset = 0;
    VkBufferView supportingAngleView = VK_NULL_HANDLE;

    VkBuffer halfedgeBuffer = VK_NULL_HANDLE;
    VkDeviceSize halfedgeOffset = 0;
    VkBufferView halfedgeView = VK_NULL_HANDLE;

    VkBuffer edgeBuffer = VK_NULL_HANDLE;
    VkDeviceSize edgeOffset = 0;
    VkBufferView edgeView = VK_NULL_HANDLE;

    VkBuffer triangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize triangleOffset = 0;
    VkBufferView triangleView = VK_NULL_HANDLE;

    VkBuffer lengthBuffer = VK_NULL_HANDLE;
    VkDeviceSize lengthOffset = 0;
    VkBufferView lengthView = VK_NULL_HANDLE;

    VkBuffer inputHalfedgeBuffer = VK_NULL_HANDLE;
    VkDeviceSize inputHalfedgeOffset = 0;
    VkBufferView inputHalfedgeView = VK_NULL_HANDLE;

    VkBuffer inputEdgeBuffer = VK_NULL_HANDLE;
    VkDeviceSize inputEdgeOffset = 0;
    VkBufferView inputEdgeView = VK_NULL_HANDLE;

    VkBuffer inputTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize inputTriangleOffset = 0;
    VkBufferView inputTriangleView = VK_NULL_HANDLE;

    VkBuffer inputLengthBuffer = VK_NULL_HANDLE;
    VkDeviceSize inputLengthOffset = 0;
    VkBufferView inputLengthView = VK_NULL_HANDLE;

    HashValues hashes{};

    bool isValid() const {
        return !geometryPositions.empty() &&
            !geometryTriangleIndices.empty() &&
            !surfacePositions.empty() &&
            surfaceNormals.size() == surfacePositions.size() &&
            !surfaceTriangleIndices.empty() &&
            surfaceTriangleIndices.size() % 3u == 0u &&
            surfaceFaceIds.size() == surfaceTriangleIndices.size() / 3u &&
            surfaceVertexIds.size() == surfacePositions.size() &&
            std::is_sorted(surfaceFaceIds.begin(), surfaceFaceIds.end()) &&
            std::adjacent_find(surfaceFaceIds.begin(), surfaceFaceIds.end()) ==
                surfaceFaceIds.end() &&
            std::is_sorted(surfaceVertexIds.begin(), surfaceVertexIds.end()) &&
            std::adjacent_find(surfaceVertexIds.begin(), surfaceVertexIds.end()) ==
                surfaceVertexIds.end() &&
            !surfaceEdgeIds.empty() &&
            std::is_sorted(surfaceEdgeIds.begin(), surfaceEdgeIds.end()) &&
            std::adjacent_find(surfaceEdgeIds.begin(), surfaceEdgeIds.end()) ==
                surfaceEdgeIds.end() &&
            intrinsicTriangleBuffer != VK_NULL_HANDLE &&
            intrinsicVertexBuffer != VK_NULL_HANDLE &&
            faceIdBuffer != VK_NULL_HANDLE &&
            faceIdView != VK_NULL_HANDLE &&
            supportingHalfedgeBuffer != VK_NULL_HANDLE &&
            supportingAngleBuffer != VK_NULL_HANDLE &&
            halfedgeBuffer != VK_NULL_HANDLE &&
            edgeBuffer != VK_NULL_HANDLE &&
            triangleBuffer != VK_NULL_HANDLE &&
            lengthBuffer != VK_NULL_HANDLE &&
            inputHalfedgeBuffer != VK_NULL_HANDLE &&
            inputEdgeBuffer != VK_NULL_HANDLE &&
            inputTriangleBuffer != VK_NULL_HANDLE &&
            inputLengthBuffer != VK_NULL_HANDLE &&
            supportingHalfedgeView != VK_NULL_HANDLE &&
            supportingAngleView != VK_NULL_HANDLE &&
            halfedgeView != VK_NULL_HANDLE &&
            edgeView != VK_NULL_HANDLE &&
            triangleView != VK_NULL_HANDLE &&
            lengthView != VK_NULL_HANDLE &&
            inputHalfedgeView != VK_NULL_HANDLE &&
            inputEdgeView != VK_NULL_HANDLE &&
            inputTriangleView != VK_NULL_HANDLE &&
            inputLengthView != VK_NULL_HANDLE;
    }

};

// Runtime view owned by VoronoiSystem. GPU buffers are borrowed; CPU topology
// arrays are generation-owned output and remain valid while this product is published.
struct VoronoiProduct {
    uint32_t candidateNodeCount = 0;
    VkBuffer candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateNeighborIndicesBufferOffset = 0;
    VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedPositionBufferOffset = 0;
    VkBuffer candidatePointVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidatePointVertexBufferOffset = 0;

    VkBuffer globalSeedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize globalSeedIdGridBufferOffset = 0;

    // Generation-owned flat topology published as borrowed views.
    const std::vector<uint32_t>* fragmentInstanceIds = nullptr;
    const std::vector<uint32_t>* fragmentSeedIds = nullptr;
    const std::vector<glm::dvec3>* fragmentCentroids = nullptr;
    const std::vector<double>* fragmentVolumes = nullptr;

    const std::vector<uint32_t>* faceFragmentA = nullptr;
    const std::vector<uint32_t>* faceFragmentB = nullptr;
    const std::vector<double>* faceAreasA = nullptr;
    const std::vector<double>* faceAreasB = nullptr;
    const std::vector<glm::dvec3>* faceAreaVectorsA = nullptr;
    const std::vector<glm::dvec3>* faceAreaVectorsB = nullptr;
    const std::vector<glm::dvec3>* faceMomentsA = nullptr;
    const std::vector<glm::dvec3>* faceMomentsB = nullptr;
    const std::vector<uint8_t>* faceTypes = nullptr;
    const std::vector<uint32_t>* faceDomainFaces = nullptr;

    const std::vector<uint32_t>* boundaryFaceIndices = nullptr;
    const std::vector<uint32_t>* faceLoopOffsets = nullptr;
    const std::vector<uint32_t>* loopVertexOffsets = nullptr;
    const std::vector<glm::dvec3>* loopVertices = nullptr;

    const std::vector<uint32_t>* contactCouplingFaceA = nullptr;
    const std::vector<uint32_t>* contactCouplingFragmentA = nullptr;
    const std::vector<uint32_t>* contactCouplingFaceB = nullptr;
    const std::vector<uint32_t>* contactCouplingFragmentB = nullptr;
    const std::vector<double>* contactCouplingAreas = nullptr;

    const std::vector<glm::vec3>* seedPositions = nullptr;
    const std::vector<uint32_t>* cellFragmentOffsets = nullptr;
    const std::vector<uint32_t>* cellFragments = nullptr;
    const std::vector<uint32_t>* instanceFragmentCounts = nullptr;

    uint32_t spaceFragmentCount = 0;
    const std::vector<uint32_t>* fragmentToSpaceIndex = nullptr;
    const std::vector<uint32_t>* spaceToFragmentIndex = nullptr;
    const std::vector<uint32_t>* seedToSpaceIndex = nullptr;
    bool hasUniqueSeedToSpaceMapping = false;

    glm::vec3 domainMin{0.0f};
    glm::vec3 domainMax{0.0f};
    glm::vec3 volumeGridMin{0.0f};
    glm::ivec3 volumeGridDim{0, 0, 0};
    float volumeGridCellSize = 0.0f;
    std::array<glm::vec3, 8> globalDomainCorners{};

    std::vector<uint32_t> globalDisplayRuntimeModelIds;
    std::vector<VkBuffer> globalDisplayVertexBuffers;
    std::vector<VkDeviceSize> globalDisplayVertexBufferOffsets;
    std::vector<VkBuffer> globalDisplayFaceIndexBuffers;
    std::vector<VkDeviceSize> globalDisplayFaceIndexBufferOffsets;
    std::vector<VkBuffer> globalDisplayCandidateBuffers;
    std::vector<VkDeviceSize> globalDisplayCandidateBufferOffsets;

    HashValues hashes{};

    bool isValid() const {
        if (fragmentInstanceIds == nullptr || fragmentSeedIds == nullptr ||
            fragmentCentroids == nullptr || fragmentVolumes == nullptr ||
            faceFragmentA == nullptr || faceFragmentB == nullptr ||
            faceAreasA == nullptr || faceAreasB == nullptr ||
            faceAreaVectorsA == nullptr || faceAreaVectorsB == nullptr ||
            faceMomentsA == nullptr || faceMomentsB == nullptr ||
            faceTypes == nullptr || faceDomainFaces == nullptr ||
            boundaryFaceIndices == nullptr || faceLoopOffsets == nullptr ||
            loopVertexOffsets == nullptr || loopVertices == nullptr ||
            contactCouplingFaceA == nullptr || contactCouplingFragmentA == nullptr ||
            contactCouplingFaceB == nullptr || contactCouplingFragmentB == nullptr ||
            contactCouplingAreas == nullptr || seedPositions == nullptr ||
            cellFragmentOffsets == nullptr || cellFragments == nullptr ||
            instanceFragmentCounts == nullptr) {
            return false;
        }
        const size_t fragmentCount = fragmentInstanceIds->size();
        const size_t faceCount = faceFragmentA->size();
        if (fragmentCount == 0 || faceCount == 0 ||
            fragmentSeedIds->size() != fragmentCount ||
            fragmentCentroids->size() != fragmentCount ||
            fragmentVolumes->size() != fragmentCount ||
            faceFragmentB->size() != faceCount ||
            faceAreasA->size() != faceCount || faceAreasB->size() != faceCount ||
            faceAreaVectorsA->size() != faceCount || faceAreaVectorsB->size() != faceCount ||
            faceMomentsA->size() != faceCount || faceMomentsB->size() != faceCount ||
            faceTypes->size() != faceCount || faceDomainFaces->size() != faceCount) {
            return false;
        }
        if (instanceFragmentCounts->size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayVertexBuffers.size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayVertexBufferOffsets.size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayFaceIndexBuffers.size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayFaceIndexBufferOffsets.size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayCandidateBuffers.size() != globalDisplayRuntimeModelIds.size() ||
            globalDisplayCandidateBufferOffsets.size() != globalDisplayRuntimeModelIds.size()) {
            return false;
        }
        if (faceLoopOffsets->size() != faceCount + 1u ||
            loopVertexOffsets->empty() ||
            loopVertexOffsets->size() != size_t(faceLoopOffsets->back()) + 1u ||
            loopVertices->size() != loopVertexOffsets->back()) {
            return false;
        }
        if (cellFragmentOffsets->size() != seedPositions->size() + 1u ||
            cellFragmentOffsets->back() != cellFragments->size()) {
            return false;
        }
        if (fragmentToSpaceIndex == nullptr || spaceToFragmentIndex == nullptr ||
            fragmentToSpaceIndex->size() != fragmentCount ||
            spaceToFragmentIndex->size() != spaceFragmentCount) {
            return false;
        }
        if (hasUniqueSeedToSpaceMapping) {
            if (seedToSpaceIndex == nullptr ||
                seedToSpaceIndex->size() != seedPositions->size()) {
                return false;
            }
        } else if (seedToSpaceIndex != nullptr) {
            return false;
        }
        if (!(volumeGridDim.x >= 2 && volumeGridDim.y >= 2 && volumeGridDim.z >= 2) ||
            !(volumeGridCellSize > 0.0f)) {
            return false;
        }
        return true;
    }
};

// Borrowed view, owned by PointComputeRuntime
struct PointProduct {
    VkBuffer positionBuffer = VK_NULL_HANDLE;
    VkDeviceSize positionBufferOffset = 0;
    uint32_t pointCount = 0;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    HashValues hashes{};

    bool isValid() const {
        return positionBuffer != VK_NULL_HANDLE && pointCount != 0;
    }
};

// Borrowed view, surface buffers owned by HeatModelRuntime and volume view by HeatSystem
struct HeatProduct {
    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<VkBuffer> modelSurfaceBuffers;
    std::vector<VkDeviceSize> modelSurfaceBufferOffsets;
    std::vector<uint32_t> modelSurfacePointCounts;
    std::vector<VkBuffer> modelSurfaceGradientBuffers;
    std::vector<VkDeviceSize> modelSurfaceGradientBufferOffsets;

    VkImageView fluidTemperatureVolumeView = VK_NULL_HANDLE;
    VkSampler fluidTemperatureVolumeSampler = VK_NULL_HANDLE;
    bool hasFluidVolume = false;

    HashValues hashes{};

    bool isValid() const {
        return !modelRuntimeModelIds.empty() &&
            modelRuntimeModelIds.size() == modelSurfaceBuffers.size() &&
            modelRuntimeModelIds.size() == modelSurfaceBufferOffsets.size() &&
            modelRuntimeModelIds.size() == modelSurfacePointCounts.size() &&
            modelRuntimeModelIds.size() == modelSurfaceGradientBuffers.size() &&
            modelRuntimeModelIds.size() == modelSurfaceGradientBufferOffsets.size();
    }
};

// Borrowed view, owned by CFDSystem
struct CFDProduct {
    uint32_t fluidNodeCount = 0;
    std::vector<uint32_t> globalFragmentToFluidNode;

    VkBuffer faceFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize faceFluxBufferSize = 0;
    VkDeviceSize faceFluxBufferOffset = 0;
    uint32_t faceFluxCount = 0;

    VkBuffer boundaryFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize boundaryFluxBufferSize = 0;
    VkDeviceSize boundaryFluxBufferOffset = 0;
    uint32_t boundaryFluxCount = 0;

    VkBuffer streamlineVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize streamlineVertexBufferOffset = 0;
    uint32_t streamlineVertexCount = 0;

    VkBuffer velocityBuffer = VK_NULL_HANDLE;
    VkDeviceSize velocityBufferOffset = 0;
    VkBuffer gradientBuffer = VK_NULL_HANDLE;
    VkDeviceSize gradientBufferOffset = 0;
    VkBuffer centroidBuffer = VK_NULL_HANDLE;
    VkDeviceSize centroidBufferOffset = 0;

    HashValues hashes{};

    bool isValid() const {
        return fluidNodeCount != 0 &&
            (faceFluxCount == 0 || faceFluxBuffer != VK_NULL_HANDLE) &&
            (boundaryFluxCount == 0 || boundaryFluxBuffer != VK_NULL_HANDLE);
    }
};
