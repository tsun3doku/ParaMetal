#pragma once

#include "hash/HashBuilder.hpp"
#include "mesh/MeshElementType.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/mat4x4.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

class IntrinsicRenderer;
struct MeshSelection;
struct SubselectionEditState;

class RemeshDisplayController {
public:
    struct SelectionConfig {
        MeshElementType elementType = MeshElementType::Face;
        uint64_t surfaceIdentity = 0;
        uint64_t revision = 0;
        size_t selectedCount = 0;
        std::vector<uint32_t> elementMask;
    };

    struct Config {
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
        VkBufferView faceIdView = VK_NULL_HANDLE;
        uint64_t surfaceIdentity = 0;
        uint64_t displayHash = 0;

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
                inputLengthView != VK_NULL_HANDLE &&
                faceIdView != VK_NULL_HANDLE &&
                surfaceIdentity != 0;
        }
    };

    void setIntrinsicRenderer(IntrinsicRenderer* updatedIntrinsicRenderer);
    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void applySelection(
        uint64_t socketKey,
        const SubselectionEditState& state,
        const MeshSelection& selection,
        uint64_t revision);
    void applyElementSelection(uint64_t socketKey, const SelectionConfig& config);
    void removeSelection(uint64_t socketKey);
    bool surfaceMatches(uint64_t socketKey, uint64_t surfaceIdentity) const;
    bool findByRuntimeModelId(
        uint32_t runtimeModelId,
        uint64_t& outSocketKey,
        uint64_t& outSurfaceIdentity) const;
    std::vector<uint32_t> registeredRuntimeModelIds() const;

private:
    IntrinsicRenderer* intrinsicRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    uint64_t selectionSocketKey = 0;
    uint64_t selectionSurfaceIdentity = 0;
    uint64_t selectionRevision = UINT64_MAX;
    MeshElementType selectionElementType = MeshElementType::Face;
};

inline uint64_t buildDisplayHash(const RemeshDisplayController::Config& config, uint64_t productDisplayHash) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showFaceNormals ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showVertexNormals ? 1u : 0u));
    HashBuilder::combinePod(hash, config.normalLength);
    HashBuilder::combinePod(hash, config.modelMatrix);
    HashBuilder::combine(hash, config.surfaceIdentity);
    HashBuilder::combine(hash, productDisplayHash);
    return hash;
}
