#pragma once

#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"

#include <cstdint>
#include <glm/mat4x4.hpp>
#include <unordered_map>
#include <unordered_set>

class IntrinsicRenderer;

class RemeshDisplayController {
public:
    struct Config {
        bool showRemeshOverlay = false;
        bool showFaceNormals = false;
        bool showVertexNormals = false;
        float normalLength = 0.05f;
        uint32_t runtimeModelId = 0;
        VkBuffer renderVertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize renderVertexBufferOffset = 0;
        VkBuffer renderIndexBuffer = VK_NULL_HANDLE;
        VkDeviceSize renderIndexBufferOffset = 0;
        uint32_t renderIndexCount = 0;
        glm::mat4 modelMatrix{ 1.0f };
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
        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showRemeshOverlay || showFaceNormals || showVertexNormals;
        }

        bool isValid() const {
            return runtimeModelId != 0 &&
                renderVertexBuffer != VK_NULL_HANDLE &&
                renderIndexBuffer != VK_NULL_HANDLE &&
                renderIndexCount != 0 &&
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

    void setIntrinsicRenderer(IntrinsicRenderer* updatedIntrinsicRenderer);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    IntrinsicRenderer* intrinsicRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};

inline uint64_t buildDisplayHash(const RemeshDisplayController::Config& config) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showRemeshOverlay ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showFaceNormals ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showVertexNormals ? 1u : 0u));
    NodeGraphHash::combinePod(hash, config.normalLength);
    NodeGraphHash::combine(hash, config.runtimeModelId);
    NodeGraphHash::combinePod(hash, config.renderVertexBuffer);
    NodeGraphHash::combinePod(hash, config.renderVertexBufferOffset);
    NodeGraphHash::combinePod(hash, config.renderIndexBuffer);
    NodeGraphHash::combinePod(hash, config.renderIndexBufferOffset);
    NodeGraphHash::combine(hash, config.renderIndexCount);
    NodeGraphHash::combinePod(hash, config.modelMatrix);
    NodeGraphHash::combinePod(hash, config.intrinsicTriangleBuffer);
    NodeGraphHash::combinePod(hash, config.intrinsicTriangleBufferOffset);
    NodeGraphHash::combinePod(hash, config.intrinsicVertexBuffer);
    NodeGraphHash::combinePod(hash, config.intrinsicVertexBufferOffset);
    NodeGraphHash::combine(hash, config.intrinsicTriangleCount);
    NodeGraphHash::combine(hash, config.intrinsicVertexCount);
    NodeGraphHash::combinePod(hash, config.averageTriangleArea);
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
    return hash;
}
