#pragma once

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

#include "domain/GeometryData.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "nodegraph/NodeGraphHash.hpp"
#include "contact/ContactTypes.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

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
    glm::mat4 modelMatrix{ 1.0f };
    uint64_t productHash = 0;

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

struct RemeshProduct {
    uint32_t runtimeModelId = 0;
    std::vector<glm::vec3> geometryPositions;
    std::vector<uint32_t> geometryTriangleIndices;
    SupportingHalfedge::IntrinsicMesh intrinsicMesh;
    VkBuffer intrinsicTriangleBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicTriangleBufferOffset = 0;
    VkBuffer intrinsicVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize intrinsicVertexBufferOffset = 0;
    size_t intrinsicTriangleCount = 0;
    size_t intrinsicVertexCount = 0;
    float averageTriangleArea = 0.0f;
    VkBufferView supportingHalfedgeView = VK_NULL_HANDLE;
    VkBufferView supportingAngleView = VK_NULL_HANDLE;
    VkBufferView halfedgeView = VK_NULL_HANDLE;
    VkBufferView edgeView = VK_NULL_HANDLE;
    VkBufferView triangleView = VK_NULL_HANDLE;
    VkBufferView lengthView = VK_NULL_HANDLE;
    VkBufferView inputHalfedgeView = VK_NULL_HANDLE;
    VkBufferView inputEdgeView = VK_NULL_HANDLE;
    VkBufferView inputTriangleView = VK_NULL_HANDLE;
    VkBufferView inputLengthView = VK_NULL_HANDLE;
    uint64_t productHash = 0;

    bool isValid() const {
        return !geometryPositions.empty() &&
            !geometryTriangleIndices.empty() &&
            !intrinsicMesh.vertices.empty() &&
            !intrinsicMesh.indices.empty() &&
            intrinsicTriangleBuffer != VK_NULL_HANDLE &&
            intrinsicVertexBuffer != VK_NULL_HANDLE &&
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

struct VoronoiProduct {
    uint32_t nodeCount = 0;
    uint32_t simNodeCount = 0;
    const voronoi::Node* mappedVoronoiNodes = nullptr;

    VkBuffer nodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize nodeBufferOffset = 0;

    VkBuffer voronoiNeighborBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiNeighborBufferOffset = 0;

    VkBuffer voronoiNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiNeighborIndicesBufferOffset = 0;

    VkBuffer voronoiInterfaceAreasBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiInterfaceAreasBufferOffset = 0;

    VkBuffer voronoiInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiInterfaceNeighborIdsBufferOffset = 0;
    VkBuffer voronoiGMLSInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiGMLSInterfaceBufferOffset = 0;
    VkBuffer simNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize simNodeBufferOffset = 0;
    VkBuffer simGMLSInterfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize simGMLSInterfaceBufferOffset = 0;
    uint32_t simGMLSInterfaceCount = 0;

    VkBuffer voronoiSeedFlagsBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiSeedFlagsBufferOffset = 0;
    VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedPositionBufferOffset = 0;
    VkBuffer occupancyPointBuffer = VK_NULL_HANDLE;
    VkDeviceSize occupancyPointBufferOffset = 0;
    uint32_t occupancyPointCount = 0;

    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<VkBuffer> modelCandidateBuffers;
    std::vector<VkDeviceSize> modelCandidateBufferOffsets;
    std::vector<VkBuffer> modelGMLSSurfaceStencilBuffers;
    std::vector<VkDeviceSize> modelGMLSSurfaceStencilBufferOffsets;
    std::vector<VkBuffer> modelGMLSSurfaceWeightBuffers;
    std::vector<VkDeviceSize> modelGMLSSurfaceWeightBufferOffsets;
    std::vector<size_t> modelGMLSSurfaceWeightCounts;
    std::vector<VkBuffer> modelGMLSSurfaceGradientWeightBuffers;
    std::vector<VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsets;
    std::vector<size_t> modelGMLSSurfaceGradientWeightCounts;
    std::vector<std::vector<uint32_t>> modelSeedFlags;
    std::vector<std::vector<glm::vec3>> modelSeedPositions;
    std::vector<uint32_t> voronoiToSim;
    std::vector<uint32_t> simToVoronoi;

    uint64_t productHash = 0;

    bool isValid() const {
        return nodeCount != 0 &&
            simNodeCount != 0 &&
            nodeBuffer != VK_NULL_HANDLE &&
            simNodeBuffer != VK_NULL_HANDLE &&
            voronoiNeighborIndicesBuffer != VK_NULL_HANDLE &&
            seedPositionBuffer != VK_NULL_HANDLE;
    }

};

struct ContactProduct {
    ContactCoupling coupling{};
    VkBuffer contactPairBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactPairBufferOffset = 0;

    uint32_t modelARuntimeModelId = 0;
    uint32_t modelBRuntimeModelId = 0;
    std::vector<ContactLineVertex> outlineVertices;
    std::vector<ContactLineVertex> correspondenceVertices;

    uint64_t productHash = 0;

    bool isValid() const {
        return coupling.isValid() &&
            contactPairBuffer != VK_NULL_HANDLE;
    }

};

struct HeatProduct {
    bool active = false;
    bool paused = false;
    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<VkBuffer> modelSurfaceBuffers;
    std::vector<VkDeviceSize> modelSurfaceBufferOffsets;
    std::vector<uint32_t> modelSurfacePointCounts;
    std::vector<VkBuffer> modelSurfaceGradientBuffers;
    std::vector<VkDeviceSize> modelSurfaceGradientBufferOffsets;

    uint64_t productHash = 0;

    bool isValid() const {
        return !modelRuntimeModelIds.empty() &&
            modelRuntimeModelIds.size() == modelSurfaceBuffers.size() &&
            modelRuntimeModelIds.size() == modelSurfaceBufferOffsets.size() &&
            modelRuntimeModelIds.size() == modelSurfacePointCounts.size() &&
            modelRuntimeModelIds.size() == modelSurfaceGradientBuffers.size() &&
            modelRuntimeModelIds.size() == modelSurfaceGradientBufferOffsets.size();
    }

};

inline void combineVkHandle(uint64_t& hash, VkBuffer handle) {
    NodeGraphHash::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

inline void combineVkHandle(uint64_t& hash, VkBufferView handle) {
    NodeGraphHash::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

inline void combineVkHandle(uint64_t& hash, VkDeviceSize handle) {
    NodeGraphHash::combine(hash, static_cast<uint64_t>(handle));
}

inline uint64_t buildProductHash(const ModelProduct& product) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, product.runtimeModelId);
    combineVkHandle(hash, product.vertexBuffer);
    NodeGraphHash::combine(hash, product.vertexBufferOffset);
    combineVkHandle(hash, product.indexBuffer);
    NodeGraphHash::combine(hash, product.indexBufferOffset);
    NodeGraphHash::combine(hash, product.indexCount);
    combineVkHandle(hash, product.renderVertexBuffer);
    NodeGraphHash::combine(hash, product.renderVertexBufferOffset);
    combineVkHandle(hash, product.renderIndexBuffer);
    NodeGraphHash::combine(hash, product.renderIndexBufferOffset);
    NodeGraphHash::combine(hash, product.renderIndexCount);
    NodeGraphHash::combinePod(hash, product.modelMatrix);
    return hash;
}

inline uint64_t buildProductHash(const RemeshProduct& product) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, product.runtimeModelId);
    NodeGraphHash::combinePodVector(hash, product.geometryPositions);
    NodeGraphHash::combinePodVector(hash, product.geometryTriangleIndices);
    NodeGraphHash::combinePodVector(hash, product.intrinsicMesh.vertices);
    NodeGraphHash::combinePodVector(hash, product.intrinsicMesh.indices);
    NodeGraphHash::combinePodVector(hash, product.intrinsicMesh.faceIds);
    NodeGraphHash::combinePodVector(hash, product.intrinsicMesh.triangles);
    combineVkHandle(hash, product.intrinsicTriangleBuffer);
    NodeGraphHash::combine(hash, product.intrinsicTriangleBufferOffset);
    combineVkHandle(hash, product.intrinsicVertexBuffer);
    NodeGraphHash::combine(hash, product.intrinsicVertexBufferOffset);
    NodeGraphHash::combine(hash, product.intrinsicTriangleCount);
    NodeGraphHash::combine(hash, product.intrinsicVertexCount);
    NodeGraphHash::combinePod(hash, product.averageTriangleArea);
    combineVkHandle(hash, product.supportingHalfedgeView);
    combineVkHandle(hash, product.supportingAngleView);
    combineVkHandle(hash, product.halfedgeView);
    combineVkHandle(hash, product.edgeView);
    combineVkHandle(hash, product.triangleView);
    combineVkHandle(hash, product.lengthView);
    combineVkHandle(hash, product.inputHalfedgeView);
    combineVkHandle(hash, product.inputEdgeView);
    combineVkHandle(hash, product.inputTriangleView);
    combineVkHandle(hash, product.inputLengthView);
    return hash;
}

inline uint64_t buildProductHash(const VoronoiProduct& product) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, product.nodeCount);
    NodeGraphHash::combine(hash, product.simNodeCount);
    NodeGraphHash::combinePodVector(hash, product.modelRuntimeModelIds);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelSeedFlags.size()));
    for (const auto& flags : product.modelSeedFlags) {
        NodeGraphHash::combinePodVector(hash, flags);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelSeedPositions.size()));
    for (const auto& positions : product.modelSeedPositions) {
        NodeGraphHash::combinePodVector(hash, positions);
    }
    NodeGraphHash::combine(hash, product.occupancyPointCount);
    NodeGraphHash::combinePodVector(hash, product.voronoiToSim);
    NodeGraphHash::combinePodVector(hash, product.simToVoronoi);
    combineVkHandle(hash, product.nodeBuffer);
    NodeGraphHash::combine(hash, product.nodeBufferOffset);
    combineVkHandle(hash, product.voronoiNeighborBuffer);
    NodeGraphHash::combine(hash, product.voronoiNeighborBufferOffset);
    combineVkHandle(hash, product.voronoiNeighborIndicesBuffer);
    NodeGraphHash::combine(hash, product.voronoiNeighborIndicesBufferOffset);
    combineVkHandle(hash, product.voronoiInterfaceAreasBuffer);
    NodeGraphHash::combine(hash, product.voronoiInterfaceAreasBufferOffset);
    combineVkHandle(hash, product.voronoiInterfaceNeighborIdsBuffer);
    NodeGraphHash::combine(hash, product.voronoiInterfaceNeighborIdsBufferOffset);
    combineVkHandle(hash, product.voronoiGMLSInterfaceBuffer);
    NodeGraphHash::combine(hash, product.voronoiGMLSInterfaceBufferOffset);
    combineVkHandle(hash, product.simNodeBuffer);
    NodeGraphHash::combine(hash, product.simNodeBufferOffset);
    combineVkHandle(hash, product.simGMLSInterfaceBuffer);
    NodeGraphHash::combine(hash, product.simGMLSInterfaceBufferOffset);
    NodeGraphHash::combine(hash, product.simGMLSInterfaceCount);
    combineVkHandle(hash, product.voronoiSeedFlagsBuffer);
    NodeGraphHash::combine(hash, product.voronoiSeedFlagsBufferOffset);
    combineVkHandle(hash, product.seedPositionBuffer);
    NodeGraphHash::combine(hash, product.seedPositionBufferOffset);
    combineVkHandle(hash, product.occupancyPointBuffer);
    NodeGraphHash::combine(hash, product.occupancyPointBufferOffset);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelCandidateBuffers.size()));
    for (size_t i = 0; i < product.modelCandidateBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelCandidateBuffers[i]);
        NodeGraphHash::combine(hash, product.modelCandidateBufferOffsets[i]);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelGMLSSurfaceStencilBuffers.size()));
    for (size_t i = 0; i < product.modelGMLSSurfaceStencilBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelGMLSSurfaceStencilBuffers[i]);
        NodeGraphHash::combine(hash, product.modelGMLSSurfaceStencilBufferOffsets[i]);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelGMLSSurfaceWeightBuffers.size()));
    for (size_t i = 0; i < product.modelGMLSSurfaceWeightBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelGMLSSurfaceWeightBuffers[i]);
        NodeGraphHash::combine(hash, product.modelGMLSSurfaceWeightBufferOffsets[i]);
        NodeGraphHash::combine(hash, product.modelGMLSSurfaceWeightCounts[i]);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelGMLSSurfaceGradientWeightBuffers.size()));
    for (size_t i = 0; i < product.modelGMLSSurfaceGradientWeightBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelGMLSSurfaceGradientWeightBuffers[i]);
        NodeGraphHash::combine(hash, product.modelGMLSSurfaceGradientWeightBufferOffsets[i]);
        NodeGraphHash::combine(hash, product.modelGMLSSurfaceGradientWeightCounts[i]);
    }
    return hash;
}

inline uint64_t buildProductHash(const ContactProduct& product) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, product.coupling.modelARuntimeModelId);
    NodeGraphHash::combine(hash, product.coupling.modelBRuntimeModelId);
    NodeGraphHash::combinePodVector(hash, product.coupling.modelBTriangleIndices);
    NodeGraphHash::combine(hash, product.coupling.contactPairCount);
    NodeGraphHash::combinePodVector(hash, product.coupling.contactPairs);
    NodeGraphHash::combine(hash, product.modelARuntimeModelId);
    NodeGraphHash::combine(hash, product.modelBRuntimeModelId);
    NodeGraphHash::combinePodVector(hash, product.outlineVertices);
    NodeGraphHash::combinePodVector(hash, product.correspondenceVertices);
    combineVkHandle(hash, product.contactPairBuffer);
    NodeGraphHash::combine(hash, product.contactPairBufferOffset);
    return hash;
}

inline uint64_t buildProductHash(const HeatProduct& product) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.active ? 1u : 0u));
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.paused ? 1u : 0u));
    NodeGraphHash::combinePodVector(hash, product.modelRuntimeModelIds);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelSurfaceBuffers.size()));
    for (size_t i = 0; i < product.modelSurfaceBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelSurfaceBuffers[i]);
        NodeGraphHash::combine(hash, product.modelSurfaceBufferOffsets[i]);
    }
    NodeGraphHash::combinePodVector(hash, product.modelSurfacePointCounts);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(product.modelSurfaceGradientBuffers.size()));
    for (size_t i = 0; i < product.modelSurfaceGradientBuffers.size(); ++i) {
        combineVkHandle(hash, product.modelSurfaceGradientBuffers[i]);
        NodeGraphHash::combine(hash, product.modelSurfaceGradientBufferOffsets[i]);
    }
    return hash;
}
