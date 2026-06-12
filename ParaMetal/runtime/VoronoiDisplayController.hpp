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

        uint64_t bindingKey = 0;
        uint32_t runtimeModelId = 0;
        VkBuffer candidateBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateBufferOffset = 0;

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
        size_t intrinsicVertexCount = 0;

        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        VkBuffer indexBuffer = VK_NULL_HANDLE;
        VkDeviceSize indexBufferOffset = 0;
        uint32_t indexCount = 0;
        glm::mat4 modelMatrix{1.0f};

        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showVoronoi || showPoints;
        }

        bool isValid() const {
            return nodeCount != 0 &&
                nodeBuffer != VK_NULL_HANDLE &&
                seedPositionBuffer != VK_NULL_HANDLE &&
                neighborIndicesBuffer != VK_NULL_HANDLE &&
                runtimeModelId != 0;
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

inline uint64_t buildDisplayHash(const VoronoiDisplayController::Config& config) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showVoronoi ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showPoints ? 1u : 0u));
    NodeGraphHash::combine(hash, config.bindingKey);
    NodeGraphHash::combine(hash, config.runtimeModelId);
    NodeGraphHash::combinePod(hash, config.nodeBuffer);
    NodeGraphHash::combine(hash, config.nodeBufferOffset);
    NodeGraphHash::combine(hash, config.nodeCount);
    NodeGraphHash::combinePod(hash, config.seedPositionBuffer);
    NodeGraphHash::combine(hash, config.seedPositionBufferOffset);
    NodeGraphHash::combinePod(hash, config.neighborIndicesBuffer);
    NodeGraphHash::combine(hash, config.neighborIndicesBufferOffset);
    NodeGraphHash::combinePod(hash, config.occupancyPointBuffer);
    NodeGraphHash::combine(hash, config.occupancyPointBufferOffset);
    NodeGraphHash::combine(hash, config.occupancyPointCount);
    NodeGraphHash::combinePod(hash, config.candidateBuffer);
    NodeGraphHash::combine(hash, config.candidateBufferOffset);
    NodeGraphHash::combinePod(hash, config.supportingHalfedgeView);
    NodeGraphHash::combinePod(hash, config.supportingAngleView);
    NodeGraphHash::combinePod(hash, config.halfedgeView);
    NodeGraphHash::combinePod(hash, config.edgeView);
    NodeGraphHash::combinePod(hash, config.triangleView);
    NodeGraphHash::combinePod(hash, config.lengthView);
    NodeGraphHash::combinePod(hash, config.inputHalfedgeView);
    NodeGraphHash::combinePod(hash, config.inputEdgeView);
    NodeGraphHash::combinePod(hash, config.inputTriangleView);
    NodeGraphHash::combinePod(hash, config.inputLengthView);
    NodeGraphHash::combine(hash, config.intrinsicVertexCount);
    NodeGraphHash::combinePod(hash, config.vertexBuffer);
    NodeGraphHash::combine(hash, config.vertexBufferOffset);
    NodeGraphHash::combinePod(hash, config.indexBuffer);
    NodeGraphHash::combine(hash, config.indexBufferOffset);
    NodeGraphHash::combine(hash, config.indexCount);
    NodeGraphHash::combinePod(hash, config.modelMatrix);
    return hash;
}
