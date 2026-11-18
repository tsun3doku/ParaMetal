#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cfd/CFDSliceGeometry.hpp"

class VulkanDevice;
class MemoryAllocator;
class UniformBufferManager;
class CommandPool;

namespace render {

class CFDSliceRenderer {
public:
    struct RenderBinding {
        uint64_t bindingKey = 0;

        VkBuffer velocityBuffer = VK_NULL_HANDLE;
        VkDeviceSize velocityBufferOffset = 0;
        VkBuffer gradientBuffer = VK_NULL_HANDLE;
        VkDeviceSize gradientBufferOffset = 0;
        VkBuffer centroidBuffer = VK_NULL_HANDLE;
        VkDeviceSize centroidBufferOffset = 0;

        uint32_t fluidNodeCount = 0;
        glm::vec3 domainMin{0.0f};
        glm::vec3 domainMax{0.0f};

        uint32_t planeAxis = 0;
        float planeFraction = 0.5f;
        bool showWireframe = false;
        std::shared_ptr<const cfd::CFDSliceGeometrySource> sliceGeometry;
    };

    CFDSliceRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~CFDSliceRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void render(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        VkExtent2D viewportExtent,
        const std::vector<RenderBinding>& slices,
        uint32_t palette,
        float minSpeed,
        float maxSpeed);
    void cleanup();

private:
    bool createPaletteTextures();
    void destroyPaletteTextures();
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipelineLayout();
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass, bool wireframe, VkPipeline& outPipeline);

    bool isRenderable(const RenderBinding& binding) const;
    void updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex, const RenderBinding& binding);
    bool updateGeometry(uint64_t key, uint32_t frameIndex, const RenderBinding& binding,
                        uint32_t axis, double coordinate);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> descriptorPools;
    std::vector<std::vector<VkDescriptorSet>> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline fillPipeline = VK_NULL_HANDLE;
    VkPipeline wireframePipeline = VK_NULL_HANDLE;

    struct GeometryBuffers {
        VkBuffer triangleBuffer = VK_NULL_HANDLE;
        VkDeviceSize triangleOffset = 0;
        uint32_t triangleVertexCount = 0;
        VkBuffer lineBuffer = VK_NULL_HANDLE;
        VkDeviceSize lineOffset = 0;
        uint32_t lineVertexCount = 0;
        uint64_t sourceHash = 0;
        uint32_t axis = UINT32_MAX;
        double coordinate = 0.0;
        bool failed = false;
    };
    std::unordered_map<uint64_t, std::vector<GeometryBuffers>> geometryByBinding;

    VkImage infernoImage = VK_NULL_HANDLE;
    VkDeviceMemory infernoMemory = VK_NULL_HANDLE;
    VkImageView infernoView = VK_NULL_HANDLE;
    VkImage viridisImage = VK_NULL_HANDLE;
    VkDeviceMemory viridisMemory = VK_NULL_HANDLE;
    VkImageView viridisView = VK_NULL_HANDLE;
    VkImage inferno2Image = VK_NULL_HANDLE;
    VkDeviceMemory inferno2Memory = VK_NULL_HANDLE;
    VkImageView inferno2View = VK_NULL_HANDLE;
    VkImage parulaImage = VK_NULL_HANDLE;
    VkDeviceMemory parulaMemory = VK_NULL_HANDLE;
    VkImageView parulaView = VK_NULL_HANDLE;
    VkSampler paletteSampler = VK_NULL_HANDLE;

    bool initialized = false;
};

} // namespace render
