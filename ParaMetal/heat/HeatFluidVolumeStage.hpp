#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "voronoi/GlobalDomainTopology.hpp"

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class HeatFluidVolumeStage {
public:
    HeatFluidVolumeStage(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool,
        uint32_t maxFramesInFlight);
    ~HeatFluidVolumeStage();

    bool initialize(const glm::ivec3& gridDim);
    void cleanup();

    bool updateSeedToFluidNode(
        const std::vector<glm::vec3>& seeds,
        const topology::GlobalDomain& globalDomain,
        const std::vector<uint32_t>& globalFragmentToFluidNode);

    void recordCompute(
        VkCommandBuffer cmd,
        uint32_t currentFrame,
        VkBuffer fluidSeedGridBuffer,
        VkDeviceSize fluidSeedGridOffset,
        VkBuffer temperatureSourceBuffer,
        VkDeviceSize temperatureSourceOffset);

    VkImageView getTemperatureVolumeView() const { return temperatureVolumeView; }
    VkSampler getTemperatureVolumeSampler() const { return temperatureVolumeSampler; }
    VkImage getTemperatureVolumeImage() const { return temperatureVolumeImage; }
    glm::ivec3 getGridDimensions() const { return gridDimensions; }
    bool isValid() const { return ready; }

private:
    bool createPipeline();
    bool createVolumeResources(const glm::ivec3& gridDim);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 0;

    glm::ivec3 gridDimensions{0};
    VkImage temperatureVolumeImage = VK_NULL_HANDLE;
    VkDeviceMemory temperatureVolumeMemory = VK_NULL_HANDLE;
    VkImageView temperatureVolumeView = VK_NULL_HANDLE;
    VkSampler temperatureVolumeSampler = VK_NULL_HANDLE;
    VkImageLayout temperatureVolumeLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer seedToFluidNodeBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedToFluidNodeBufferOffset = 0;
    uint32_t seedToFluidNodeCount = 0;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    bool ready = false;
};
