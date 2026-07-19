#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "util/GlyphText.hpp"

#include <cstdint>
#include <vector>

class CommandPool;
class MemoryAllocator;
class VulkanDevice;

class ScreenTextRenderer {
public:
    ScreenTextRenderer(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& allocator,
        uint32_t maxFramesInFlight,
        VkRenderPass renderPass,
        uint32_t subpassIndex,
        CommandPool& commandPool);
    ~ScreenTextRenderer();

    bool isReady() const { return ready; }
    const GlyphText& getGlyphText() const { return glyphText; }

    void beginFrame(uint32_t currentFrame);
    bool draw(
        VkCommandBuffer commandBuffer,
        uint32_t currentFrame,
        VkExtent2D extent,
        const std::vector<GlyphText::GlyphInstance>& glyphs);
    void cleanup();

private:
    struct QuadVertex {
        glm::vec2 position{};
        glm::vec2 texCoord{};
    };

    bool createQuadVertexBuffer();
    bool createInstanceBuffers(uint32_t maxFramesInFlight);
    bool createFontAtlas();
    bool createDescriptors(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpassIndex);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& allocator;
    CommandPool& commandPool;

    VkBuffer quadVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize quadVertexBufferOffset = 0;
    std::vector<VkBuffer> instanceBuffers;
    std::vector<VkDeviceSize> instanceBufferOffsets;
    std::vector<void*> instanceBuffersMapped;
    std::vector<uint32_t> frameGlyphCursors;

    VkImage fontAtlasImage = VK_NULL_HANDLE;
    VkDeviceMemory fontAtlasMemory = VK_NULL_HANDLE;
    VkImageView fontAtlasView = VK_NULL_HANDLE;
    VkSampler fontSampler = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    GlyphText glyphText;
    uint32_t maxGlyphCapacity = 2048;
    bool ready = false;
};
