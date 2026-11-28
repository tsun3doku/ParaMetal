#pragma once

#include "framegraph/FrameGraphTypes.hpp"
#include "mesh/MeshElementType.hpp"

#include <vulkan/vulkan.h>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

class VulkanDevice;
class VkFrameGraphRuntime;

struct PickGeometryBinding {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize vertexOffset = 0;
    VkDeviceSize vertexRange = 0;
    uint32_t vertexCount = 0;

    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    VkDeviceSize indexRange = 0;
    uint32_t indexCount = 0;

    uint32_t runtimeModelId = 0;
    uint64_t geometryIdentity = 0;
};

struct SubselectionPickRequest {
    int x = 0;
    int y = 0;
    uint64_t sessionId = 0;
    uint64_t requestId = 0;
    SurfaceBinding binding = SurfaceBinding::Source;
    uint64_t displaySocketKey = 0;
    uint64_t pickGeometryIdentity = 0;
    uint32_t runtimeModelId = 0;
    MeshElementType elementType = MeshElementType::Face;
    glm::uvec2 viewport{0, 0};
    glm::mat4 viewProjection{1.0f};
    glm::mat4 modelMatrix{1.0f};
    bool additive = false;
};

struct SubselectionPickResult {
    uint64_t sessionId = 0;
    uint64_t requestId = 0;
    MeshElementType elementType = MeshElementType::Face;
    uint32_t elementId = 0;
    glm::uvec2 endpoints{0, 0};
    bool hit = false;
    bool cancelled = false;
    bool additive = false;
};

class SubselectionPickService {
public:
    SubselectionPickService(
        VulkanDevice& device,
        VkFrameGraphRuntime& frameGraphRuntime,
        framegraph::ResourceId pickResourceId,
        uint32_t maxFramesInFlight);
    ~SubselectionPickService();

    SubselectionPickService(const SubselectionPickService&) = delete;
    SubselectionPickService& operator=(const SubselectionPickService&) = delete;

    void setViewportExtent(uint32_t width, uint32_t height);
    void queuePickRequest(const SubselectionPickRequest& request);
    void cancelThrough(uint64_t sessionId, uint64_t requestId);
    const SubselectionPickRequest* pendingRequest(uint32_t currentFrame) const;
    bool prepareComputePick(
        uint32_t currentFrame,
        const PickGeometryBinding& binding);
    void cancelFrameRequest(uint32_t currentFrame);
    void processPickingRequests(uint32_t currentFrame);
    void recordReadback(VkCommandBuffer commandBuffer, uint32_t currentFrame);
    bool recordComputePick(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        const PickGeometryBinding& binding);
    std::optional<SubselectionPickResult> takeResult();

private:
    struct FrameSlot {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkBuffer computeResultBuffer = VK_NULL_HANDLE;
        VkDeviceMemory computeResultMemory = VK_NULL_HANDLE;
        void* computeResultMapped = nullptr;
        SubselectionPickRequest request{};
        bool requestPending = false;
        bool copyPending = false;
        bool computePending = false;
        bool computeDescriptorPrepared = false;
        bool cancelled = false;
    };

    bool createFrameSlots(uint32_t maxFramesInFlight);
    bool createFrameSlot(FrameSlot& slot);
    bool createComputeResources();
    void completeCancelledResult(uint32_t currentFrame);
    void cleanup();

    VkSampler pickSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool computeDescriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkPipelineLayout computePipelineLayout = VK_NULL_HANDLE;
    VkPipeline computePipeline = VK_NULL_HANDLE;

    VulkanDevice& vulkanDevice;
    VkFrameGraphRuntime& frameGraphRuntime;
    framegraph::ResourceId pickResourceId{};
    std::vector<FrameSlot> frameSlots;
    uint32_t viewportWidth = 0;
    uint32_t viewportHeight = 0;

    std::mutex queueMutex;
    std::queue<SubselectionPickRequest> pickQueue;
    std::deque<SubselectionPickResult> completedResults;
};
