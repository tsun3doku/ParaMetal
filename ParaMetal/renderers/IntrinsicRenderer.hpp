#pragma once

#include "runtime/RemeshDisplayController.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include <unordered_map>
#include <vector>

class VulkanDevice;
class MemoryAllocator;
class ModelRegistry;
class ModelSelection;
class UniformBufferManager;
class CommandPool;
class iODT;

class IntrinsicRenderer {
public:
    IntrinsicRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& commandPool);
    ~IntrinsicRenderer();
    void cleanup();

    void setModelRegistry(ModelRegistry* registry);
    bool apply(uint64_t socketKey, const RemeshDisplayController::Config& config);
    void remove(uint64_t socketKey);
    void applySelection(uint64_t socketKey, const RemeshDisplayController::SelectionConfig& config);
    void removeSelection(uint64_t socketKey);
    void prepareFrame(uint32_t currentFrame, glm::vec2 viewportSize);
    bool initializeGeometry(VkRenderPass renderPass, uint32_t maxFramesInFlight, uint32_t subpassIndex);
    bool initializeOverlay(VkRenderPass renderPass, uint32_t maxFramesInFlight, uint32_t subpassIndex);
    bool initializePick(VkRenderPass renderPass, uint32_t subpassIndex);
    void renderGeometry(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        int wireframeMode,
        const ModelSelection* modelSelection);
    void renderOverlay(VkCommandBuffer commandBuffer, uint32_t currentFrame);
    void renderXRayWireframe(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        const ModelSelection& modelSelection);
    void renderElementPick(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        uint64_t socketKey,
        uint64_t surfaceIdentity,
        MeshElementType elementType);
    bool surfaceMatches(uint64_t socketKey, uint64_t surfaceIdentity) const;

private:
    static constexpr float VertexHighlightRadiusWorld = 0.01f;
    static constexpr float EdgeHighlightHalfWidthPixels = 1.5f;

    struct SurfacePushConstants {
        glm::mat4 modelMatrix{1.0f};
        glm::vec4 sourceParams{0.0f};
        glm::vec4 selectionParams{0.0f};
        glm::vec4 wireframeColor{0.0f};
    };

    struct ElementMaskBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkBufferView view = VK_NULL_HANDLE;
        std::vector<uint32_t> data;
        uint64_t revision = UINT64_MAX;
    };

    uint32_t calculateMipLevels(uint32_t width, uint32_t height);
    bool createWireframeTexture();
    void releaseDescriptorSetsForSocket(uint64_t socketKey);
    bool allocateSupportingHalfedgeDescriptorSets(uint64_t socketKey, uint32_t maxFramesInFlight);
    bool allocateNormalDescriptorSets(
        uint64_t socketKey,
        uint32_t maxFramesInFlight,
        std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets);
    bool updateSupportingHalfedgeDescriptorSet(uint64_t socketKey, const RemeshDisplayController::Config& config, uint32_t currentFrame);
    bool updateNormalDescriptorSet(
        uint64_t socketKey,
        const RemeshDisplayController::Config& config,
        uint32_t currentFrame,
        VkBuffer buffer,
        VkDeviceSize bufferOffset,
        VkDeviceSize bufferRange,
        std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets);

    bool createSupportingHalfedgeDescriptorPool(uint32_t maxFramesInFlight);
    bool createSupportingHalfedgeDescriptorSetLayout();

    bool createNormalDescriptorPool(uint32_t maxFramesInFlight);
    bool createNormalDescriptorSetLayout();
    bool createNormalPipelineLayout();

    bool createSupportingHalfedgePipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    bool createNormalPipeline(
        VkRenderPass renderPass,
        uint32_t subpassIndex,
        const char* vertexShaderPath,
        const char* geometryShaderPath,
        const char* fragmentShaderPath,
        VkPipeline& outPipeline);
    bool createXrayWireframePipeline(
        VkRenderPass renderPass,
        uint32_t subpassIndex);
    void drawIntrinsicSurfaces(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        VkPipeline pipeline,
        int mode,
        bool applySelection,
        bool setStencil,
        const ModelSelection* modelSelection);
    bool createElementPickPipelines(VkRenderPass renderPass, uint32_t subpassIndex);
    bool createElementPickPipeline(
        VkRenderPass renderPass,
        uint32_t subpassIndex,
        VkShaderModule vertShaderModule,
        VkShaderModule geomShaderModule,
        const char* fragmentShaderPath,
        VkPipeline& outPipeline);
    bool ensureElementMaskBuffers(uint64_t socketKey, size_t elementCount, uint32_t currentFrame);
    VkBufferView elementMaskView(uint64_t socketKey, uint32_t currentFrame) const;
    void cleanupOverlay();
    void cleanupPick();
    void renderNormals(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        VkPipeline pipeline,
        const std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& descriptorSetsBySocket,
        bool faceNormals);
    glm::mat4 resolveModelMatrix(const RemeshDisplayController::Config& config) const;
    float resolveVertexMarkerRadius(
        const RemeshDisplayController::Config& config) const;

    VulkanDevice& vulkanDevice;
    MemoryAllocator& allocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& renderCommandPool;
    ModelRegistry* modelRegistry = nullptr;
    uint32_t maxFramesInFlight = 0;

    VkDescriptorPool supportingHalfedgeDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout supportingHalfedgeDescriptorSetLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> supportingHalfedgeDescriptorSetsBySocket;

    VkImage wireframeTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory wireframeTextureMemory = VK_NULL_HANDLE;
    VkImageView wireframeTextureView = VK_NULL_HANDLE;
    VkSampler wireframeTextureSampler = VK_NULL_HANDLE;

    VkDescriptorPool normalDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout normalDescriptorSetLayout = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> intrinsicNormalsDescriptorSetsBySocket;
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>> intrinsicVertexNormalsDescriptorSetsBySocket;
    std::unordered_map<uint64_t, RemeshDisplayController::Config> remeshConfigsBySocket;
    std::unordered_map<uint64_t, RemeshDisplayController::SelectionConfig> selectionConfigsBySocket;
    std::unordered_map<uint64_t, std::vector<ElementMaskBuffer>> elementMaskBuffersBySocket;

    VkBuffer zeroElementMaskBuffer = VK_NULL_HANDLE;
    VkDeviceSize zeroElementMaskOffset = 0;
    VkBufferView zeroElementMaskView = VK_NULL_HANDLE;
    glm::vec2 viewportSize{0.0f};

    VkPipeline supportingHalfedgePipeline = VK_NULL_HANDLE;
    VkPipelineLayout supportingHalfedgePipelineLayout = VK_NULL_HANDLE;
    VkPipeline intrinsicNormalsPipeline = VK_NULL_HANDLE;
    VkPipeline intrinsicVertexNormalsPipeline = VK_NULL_HANDLE;
    VkPipeline xrayWireframePipeline = VK_NULL_HANDLE;
    VkPipelineLayout normalPipelineLayout = VK_NULL_HANDLE;
    VkPipeline intrinsicFacePickPipeline = VK_NULL_HANDLE;
    VkPipeline intrinsicVertexPickPipeline = VK_NULL_HANDLE;
    VkPipeline intrinsicEdgePickPipeline = VK_NULL_HANDLE;

    bool surfaceInitialized = false;
    bool overlayInitialized = false;
    bool pickInitialized = false;
};
