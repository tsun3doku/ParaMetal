#pragma once

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

// Borrowed view, owned by RemeshSystem
struct RemeshProduct {
    uint32_t runtimeModelId = 0;
    std::vector<glm::vec3> geometryPositions;
    std::vector<uint32_t> geometryTriangleIndices;
    std::vector<glm::vec3> surfacePositions;
    std::vector<glm::vec3> surfaceNormals;
    std::vector<uint32_t> surfaceTriangleIndices;
    VkBuffer intrinsicTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicTriangleBufferOffset = 0;
    VkBuffer intrinsicVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicVertexBufferOffset = 0;
    size_t intrinsicTriangleCount = 0;
    size_t intrinsicVertexCount = 0;
    float averageTriangleArea = 0.0f;

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
            intrinsicTriangleBuffer != VK_NULL_HANDLE &&
            intrinsicVertexBuffer != VK_NULL_HANDLE &&
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

// Borrowed view, owned by VoronoiSystem
struct VoronoiProduct {
    uint32_t candidateNodeCount = 0;
    VkBuffer candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateNeighborIndicesBufferOffset = 0;
    VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedPositionBufferOffset = 0;
    VkBuffer candidatePointVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidatePointVertexBufferOffset = 0;
    VkBuffer contactVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactVertexBufferOffset = 0;
    uint32_t contactVertexCount = 0;

    VkBuffer globalSeedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize globalSeedIdGridBufferOffset = 0;
    uint32_t globalSeedIdGridCount = 0;

    HashValues hashes{};

    std::vector<glm::vec4> globalSeedPositions;
    std::array<glm::vec3, 8> globalDomainCorners{};
    glm::vec3 globalVolumeGridMin{};
    glm::ivec3 globalVolumeGridDim{};
    float globalVolumeGridCellSize = 0.0f;
    uint32_t globalFragmentCount = 0;
    std::vector<uint32_t> fragmentInstanceIds;
    std::vector<uint32_t> fragmentSeedIds;
    std::vector<glm::dvec3> fragmentCentroids;
    std::vector<double> fragmentVolumes;
    // Seed to fluid fragment provenance with InvalidId for empty seeds
    std::vector<uint32_t> seedToFragment;
    float nominalCellSize = 0.0f;
    uint32_t globalFaceCount = 0;
    std::vector<uint32_t> faceFragmentA;
    std::vector<uint32_t> faceFragmentB;
    std::vector<double> faceAreasA;
    std::vector<glm::dvec3> faceAreaVectorsA;
    std::vector<glm::dvec3> faceMomentsA;
    std::vector<double> faceAreasB;
    std::vector<glm::dvec3> faceAreaVectorsB;
    std::vector<glm::dvec3> faceMomentsB;
    std::vector<uint8_t> faceTypes;
    std::vector<uint32_t> faceDomainFaces;
    uint32_t globalContactCouplingCount = 0;
    std::vector<uint32_t> contactCouplingFaceA;
    std::vector<uint32_t> contactCouplingFaceB;
    std::vector<double> contactCouplingAreas;
    uint32_t globalBoundaryFaceCount = 0;
    std::vector<uint32_t> boundaryFaceIndices;
    std::vector<uint32_t> boundaryFaceFragments;
    std::vector<uint32_t> boundaryFaceDomainFaces;
    std::vector<uint32_t> instanceFragmentCounts;

    std::vector<uint32_t> globalDisplayRuntimeModelIds;
    std::vector<VkBuffer> globalDisplayVertexBuffers;
    std::vector<VkDeviceSize> globalDisplayVertexBufferOffsets;
    std::vector<VkBuffer> globalDisplayFaceIndexBuffers;
    std::vector<VkDeviceSize> globalDisplayFaceIndexBufferOffsets;
    std::vector<VkBuffer> globalDisplayCandidateBuffers;
    std::vector<VkDeviceSize> globalDisplayCandidateBufferOffsets;

    bool isValid() const {
        return !globalSeedPositions.empty() &&
            globalVolumeGridDim.x >= 2 && globalVolumeGridDim.y >= 2 && globalVolumeGridDim.z >= 2 &&
            fragmentInstanceIds.size() == globalFragmentCount &&
            fragmentSeedIds.size() == globalFragmentCount &&
            fragmentCentroids.size() == globalFragmentCount &&
            fragmentVolumes.size() == globalFragmentCount &&
            nominalCellSize > 0.0f &&
            faceFragmentA.size() == globalFaceCount &&
            faceFragmentB.size() == globalFaceCount &&
            faceAreasA.size() == globalFaceCount &&
            faceAreaVectorsA.size() == globalFaceCount &&
            faceMomentsA.size() == globalFaceCount &&
            faceAreasB.size() == globalFaceCount &&
            faceAreaVectorsB.size() == globalFaceCount &&
            faceMomentsB.size() == globalFaceCount &&
            faceTypes.size() == globalFaceCount &&
            faceDomainFaces.size() == globalFaceCount &&
            contactCouplingFaceA.size() == globalContactCouplingCount &&
            contactCouplingFaceB.size() == globalContactCouplingCount &&
            contactCouplingAreas.size() == globalContactCouplingCount &&
            boundaryFaceIndices.size() == globalBoundaryFaceCount &&
            boundaryFaceFragments.size() == globalBoundaryFaceCount &&
            boundaryFaceDomainFaces.size() == globalBoundaryFaceCount &&
            !instanceFragmentCounts.empty();
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

    VkBuffer velocityBuffer = VK_NULL_HANDLE;
    VkDeviceSize velocityBufferSize = 0;
    VkDeviceSize velocityBufferOffset = 0;

    VkBuffer faceFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize faceFluxBufferSize = 0;
    VkDeviceSize faceFluxBufferOffset = 0;
    uint32_t faceFluxCount = 0;

    VkBuffer boundaryFluxBuffer = VK_NULL_HANDLE;
    VkDeviceSize boundaryFluxBufferSize = 0;
    VkDeviceSize boundaryFluxBufferOffset = 0;
    uint32_t boundaryFluxCount = 0;

    HashValues hashes{};

    bool isValid() const {
        return fluidNodeCount != 0 &&
            velocityBuffer != VK_NULL_HANDLE &&
            (faceFluxCount == 0 || faceFluxBuffer != VK_NULL_HANDLE) &&
            (boundaryFluxCount == 0 || boundaryFluxBuffer != VK_NULL_HANDLE);
    }
};

