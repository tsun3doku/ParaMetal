#pragma once

#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace render {
class VoronoiOverlayRenderer;
}

class VoronoiDisplayController {
public:
    struct Config {
        bool showVoronoi = false;
        bool showPoints = false;
        uint32_t nodeCount = 0;
        const voronoi::Node* mappedVoronoiNodes = nullptr;
        VkBuffer nodeBuffer = VK_NULL_HANDLE;
        VkDeviceSize nodeBufferOffset = 0;
        VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
        VkDeviceSize seedPositionBufferOffset = 0;
        VkBuffer neighborIndicesBuffer = VK_NULL_HANDLE;
        VkDeviceSize neighborIndicesBufferOffset = 0;
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
        std::vector<VkBuffer> modelGMLSSurfaceGradientWeightBuffers;
        std::vector<VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsets;

        std::vector<VkBufferView> modelSupportingHalfedgeViews;
        std::vector<VkBufferView> modelSupportingAngleViews;
        std::vector<VkBufferView> modelHalfedgeViews;
        std::vector<VkBufferView> modelEdgeViews;
        std::vector<VkBufferView> modelTriangleViews;
        std::vector<VkBufferView> modelLengthViews;
        std::vector<VkBufferView> modelInputHalfedgeViews;
        std::vector<VkBufferView> modelInputEdgeViews;
        std::vector<VkBufferView> modelInputTriangleViews;
        std::vector<VkBufferView> modelInputLengthViews;
        std::vector<size_t> modelIntrinsicVertexCounts;

        std::vector<VkBuffer> modelVertexBuffers;
        std::vector<VkDeviceSize> modelVertexBufferOffsets;
        std::vector<VkBuffer> modelIndexBuffers;
        std::vector<VkDeviceSize> modelIndexBufferOffsets;
        std::vector<uint32_t> modelIndexCounts;
        std::vector<glm::mat4> modelMatrices;

        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showVoronoi || showPoints;
        }

        bool isValid() const {
            return nodeCount != 0 &&
                nodeBuffer != VK_NULL_HANDLE &&
                seedPositionBuffer != VK_NULL_HANDLE &&
                neighborIndicesBuffer != VK_NULL_HANDLE;
        }

    };

    void setOverlayRenderer(render::VoronoiOverlayRenderer* updatedOverlayRenderer);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    render::VoronoiOverlayRenderer* overlayRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};

inline uint64_t buildDisplayHash(const VoronoiDisplayController::Config& config, uint64_t productHash) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showVoronoi ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showPoints ? 1u : 0u));
    NodeGraphHash::combine(hash, productHash);
    NodeGraphHash::combinePodVector(hash, config.modelRuntimeModelIds);
    NodeGraphHash::combinePodVector(hash, config.modelCandidateBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelCandidateBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceStencilBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceStencilBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceWeightBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceWeightBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceGradientWeightBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelGMLSSurfaceGradientWeightBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelSupportingHalfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.modelSupportingAngleViews);
    NodeGraphHash::combinePodVector(hash, config.modelHalfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.modelEdgeViews);
    NodeGraphHash::combinePodVector(hash, config.modelTriangleViews);
    NodeGraphHash::combinePodVector(hash, config.modelLengthViews);
    NodeGraphHash::combinePodVector(hash, config.modelInputHalfedgeViews);
    NodeGraphHash::combinePodVector(hash, config.modelInputEdgeViews);
    NodeGraphHash::combinePodVector(hash, config.modelInputTriangleViews);
    NodeGraphHash::combinePodVector(hash, config.modelInputLengthViews);
    NodeGraphHash::combinePodVector(hash, config.modelIntrinsicVertexCounts);
    NodeGraphHash::combinePodVector(hash, config.modelVertexBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelVertexBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelIndexBuffers);
    NodeGraphHash::combinePodVector(hash, config.modelIndexBufferOffsets);
    NodeGraphHash::combinePodVector(hash, config.modelIndexCounts);
    NodeGraphHash::combinePodVector(hash, config.modelMatrices);
    return hash;
}
