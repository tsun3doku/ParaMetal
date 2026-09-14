#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <vector>

class VulkanDevice;
class MemoryAllocator;
class UniformBufferManager;
class CommandPool;

namespace render {

class HeatVolumeRenderer {
public:
    struct VolumePushConstants {
        glm::mat4 worldToDomain{1.0f};
        glm::vec3 gridMin{0.0f};
        float cellSize = 0.0f;
        glm::ivec3 gridDim{0};
        float opacity = 0.25f;
        float minTemperature = 0.0f;
        float maxTemperature = 100.0f;
        uint32_t palette = 0;
    };

    HeatVolumeRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uniformBufferManager, CommandPool& commandPool);
    ~HeatVolumeRenderer();

    bool initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight);
    void cleanup();

    void setEnabled(bool isEnabled) { enabled = isEnabled; }
    bool isEnabled() const { return enabled; }

    void setOpacity(float op) { opacity = glm::clamp(op, 0.0f, 1.0f); }
    float getOpacity() const { return opacity; }

    void setPalette(uint32_t pal) { palette = pal < 4 ? pal : 0; }
    uint32_t getPalette() const { return palette; }

    void setRange(float minTemp, float maxTemp) { minTemperature = minTemp; maxTemperature = maxTemp; }

    void setVolumeTexture(VkImageView view, VkSampler sampler, const glm::vec3& gridMin, const glm::ivec3& gridDim, float cellSize);
    void setVolumeTexture(VkImageView view, VkSampler sampler);
    void setVolumeGrid(const glm::vec3& gridMin, const glm::ivec3& gridDim, float cellSize);
    void setVolumeDomain(const std::array<glm::vec3, 8>& corners);

    void render(VkCommandBuffer cmd, uint32_t frameIndex);

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool(uint32_t maxFramesInFlight);
    bool createDescriptorSets(uint32_t maxFramesInFlight);
    bool createPipeline(VkRenderPass renderPass, uint32_t subpass);
    bool createPaletteTextures();
    void destroyPaletteTextures();

    void updateStaticDescriptorSets();
    void updateVolumeDescriptorSets();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

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

    VkImageView volumeImageView = VK_NULL_HANDLE;
    VkSampler volumeSampler = VK_NULL_HANDLE;
    glm::vec3 volumeGridMin{0.0f};
    glm::ivec3 volumeGridDim{0};
    float volumeCellSize = 0.0f;

    glm::mat4 worldToDomain{1.0f};
    bool hasDomain = false;

    bool enabled = false;
    float opacity = 0.25f;
    uint32_t palette = 0;
    float minTemperature = 0.0f;
    float maxTemperature = 100.0f;

    uint32_t maxFramesInFlight = 0;
    bool initialized = false;
};

} // namespace render
