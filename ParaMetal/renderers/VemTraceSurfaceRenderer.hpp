#pragma once

#include "voronoi/GlobalDomainTopology.hpp"

#include <vulkan/vulkan.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

class VulkanDevice;
class UniformBufferManager;
class MemoryAllocator;
class CommandPool;

class VemTraceSurfaceRenderer {
public:
    struct PushConstant {
        alignas(16) glm::mat4 modelMatrix{1.0f};
        alignas(16) glm::vec4 sourceParams{0.0f};
        uint32_t palette = 0;
        float minTemperature = 0.0f;
        float maxTemperature = 100.0f;
        float isocontourSpacing = 0.0f;
        uint32_t traceCount = 0;
        uint32_t padding[3] = {};
    };

    struct PaletteAssets {
        VkSampler sampler = VK_NULL_HANDLE;
        VkImageView inferno = VK_NULL_HANDLE;
        VkImageView viridis = VK_NULL_HANDLE;
        VkImageView inferno2 = VK_NULL_HANDLE;
        VkImageView parula = VK_NULL_HANDLE;
        VkImageView isocontour = VK_NULL_HANDLE;
        VkSampler isocontourSampler = VK_NULL_HANDLE;
    };

    struct RenderBinding {
        uint32_t runtimeModelId = 0;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        uint32_t vertexCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        VkBuffer traceTemperatureBuffer = VK_NULL_HANDLE;
        uint32_t traceCount = 0;
    };

    VemTraceSurfaceRenderer(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool);
    ~VemTraceSurfaceRenderer();

    void initialize(
        VkRenderPass renderPass,
        uint32_t subpass,
        uint32_t maxFramesInFlight,
        const PaletteAssets& paletteAssets);
    void cleanup();

    void setPalette(uint32_t updatedPalette) {
        palette = updatedPalette < 4 ? updatedPalette : 0u;
    }
    void setRange(float minimum, float maximum) {
        minTemperature = minimum;
        maxTemperature = maximum;
    }
    void setIsocontourSpacing(float spacing) {
        isocontourSpacing = spacing;
    }
    uint32_t getPalette() const { return palette; }
    float getMinTemperature() const { return minTemperature; }
    float getMaxTemperature() const { return maxTemperature; }
    float getIsocontourSpacing() const { return isocontourSpacing; }

    void render(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const std::vector<RenderBinding>& surfaces);

private:
    struct CachedDescriptor {
        uint32_t frameIndex = 0;
        uint32_t runtimeModelId = 0;
        VkBuffer traceBuffer = VK_NULL_HANDLE;
        uint32_t traceCount = 0;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    bool createDescriptorPool(uint32_t maxSets);
    bool ensureDescriptorPool();
    VkDescriptorSet allocateDescriptorSet(VkDescriptorPool pool);
    bool createDescriptorSetLayout();
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);
    VkDescriptorSet findOrCreateDescriptorSet(
        uint32_t frameIndex,
        const RenderBinding& binding);
    void updateDescriptorSet(
        VkDescriptorSet descriptorSet,
        uint32_t frameIndex,
        const RenderBinding& binding);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;

    PaletteAssets paletteAssets{};
    std::vector<VkDescriptorPool> descriptorPools;
    std::vector<CachedDescriptor> cachedDescriptors;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool initialized = false;
    uint32_t palette = 0;
    float minTemperature = 0.0f;
    float maxTemperature = 100.0f;
    float isocontourSpacing = 0.0f;
};

static_assert(offsetof(VemTraceSurfaceRenderer::PushConstant, palette) == 80);
static_assert(offsetof(VemTraceSurfaceRenderer::PushConstant, minTemperature) == 84);
static_assert(offsetof(VemTraceSurfaceRenderer::PushConstant, maxTemperature) == 88);
static_assert(offsetof(VemTraceSurfaceRenderer::PushConstant, isocontourSpacing) == 92);
static_assert(offsetof(VemTraceSurfaceRenderer::PushConstant, traceCount) == 96);
static_assert(sizeof(VemTraceSurfaceRenderer::PushConstant) == 112);
