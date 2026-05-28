#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include "util/GlyphText.hpp"

#include <cstdint>
#include <string>
#include <vector>

class VulkanDevice;
class CommandPool;

class HeatPaletteRenderer {
public:
    HeatPaletteRenderer(VulkanDevice& vulkanDevice, CommandPool& commandPool);
    ~HeatPaletteRenderer();

    void initialize(VkRenderPass renderPass, uint32_t subpassIndex, uint32_t maxFramesInFlight);
    void setVisible(bool visible);
    void setRange(float minTemp, float maxTemp);
    void render(VkCommandBuffer commandBuffer, uint32_t currentFrame, VkExtent2D extent);
    void cleanup();

private:
    struct BarPushConstants {
        glm::vec4 barRect;
        glm::vec2 viewportSize;
        float minTemp;
        float maxTemp;
    };

    void createBarPipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    void createTextPipeline(VkRenderPass renderPass, uint32_t subpassIndex);
    void createQuadVertexBuffer();
    void createTextInstanceBuffers(uint32_t maxFramesInFlight);
    void createFontAtlas();
    void createTextDescriptorPool(uint32_t maxFramesInFlight);
    void createTextDescriptorSetLayout();
    void createTextDescriptorSets(uint32_t maxFramesInFlight);
    void buildGlyphInstances(VkExtent2D extent);

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 0;

    VkExtent2D viewportExtent = {};
    bool visible = false;
    float minTemp = 0.0f;
    float maxTemp = 100.0f;

    VkPipelineLayout barPipelineLayout = VK_NULL_HANDLE;
    VkPipeline barPipeline = VK_NULL_HANDLE;

    VkBuffer quadVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory quadVertexBufferMemory = VK_NULL_HANDLE;

    std::vector<VkBuffer> textInstanceBuffers;
    std::vector<VkDeviceMemory> textInstanceBufferMemories;
    std::vector<void*> textInstanceBuffersMapped;

    VkImage fontAtlasImage = VK_NULL_HANDLE;
    VkDeviceMemory fontAtlasMemory = VK_NULL_HANDLE;
    VkImageView fontAtlasView = VK_NULL_HANDLE;
    VkSampler fontSampler = VK_NULL_HANDLE;

    VkDescriptorPool textDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout textDescriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> textDescriptorSets;

    VkPipelineLayout textPipelineLayout = VK_NULL_HANDLE;
    VkPipeline textPipeline = VK_NULL_HANDLE;

    GlyphText glyphText;
    std::vector<GlyphText::GlyphInstance> glyphInstances;
    uint32_t glyphCount = 0;
    uint32_t maxGlyphCapacity = 256;
    bool glyphInstancesDirty = true;
};
