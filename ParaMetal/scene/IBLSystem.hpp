#pragma once

#include <string>
#include <vulkan/vulkan.h>

class CommandPool;
class MemoryAllocator;
class VulkanDevice;

class IBLSystem {
public:
    static constexpr const char* DefaultEnvironmentPath = "textures/ibl/default.exr";

    IBLSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);
    ~IBLSystem();

    bool initialize(const std::string& hdrPath = DefaultEnvironmentPath);
    void cleanup();
    bool isInitialized() const { return initialized; }

    VkImageView getIrradianceView() const { return irradianceCubeView; }
    VkImageView getPrefilteredView() const { return prefilteredCubeView; }
    VkImageView getBrdfLutView() const { return brdfLutView; }
    VkSampler getSampler() const { return iblSampler; }
    VkSampler getEquirectSampler() const { return equirectSampler; }

private:
    static constexpr uint32_t EnvironmentCubeSize = 512;
    static constexpr uint32_t EnvironmentMipLevels = 10; // floor(log2(512)) + 1
    static constexpr uint32_t IrradianceCubeSize = 64;
    static constexpr uint32_t PrefilteredCubeSize = 256;
    static constexpr uint32_t PrefilteredMipLevels = 7;
    static constexpr uint32_t BrdfLutSize = 512;
    static constexpr uint32_t ComputeLocalSize = 8;

    struct EquirectPC {
        uint32_t faceSize;
    };

    struct IrradiancePC {
        uint32_t faceSize;
    };

    struct PrefilterPC {
        uint32_t faceSize;
        uint32_t mipLevel;
        uint32_t mipCount;
        uint32_t envMapSize;
    };

    struct BrdfPC {
        uint32_t lutSize;
    };

    bool loadEquirectangularHDR(const std::string& hdrPath);
    bool createImages();
    bool createSampler();
    bool createEquirectSampler();
    bool generateIBLMaps();

    bool create2DImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory);
    bool createCubeImage(uint32_t size, uint32_t mipLevels, VkFormat format, VkImage& image, VkDeviceMemory& memory);
    bool createCubeImage(uint32_t size, uint32_t mipLevels, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory);
    bool create2DView(VkImage image, VkFormat format, VkImageView& imageView);
    bool createCubeView(VkImage image, VkFormat format, uint32_t mipLevels, VkImageView& imageView);
    bool createStorageView(VkImage image, VkImageViewType viewType, VkFormat format, uint32_t mipLevel, uint32_t layerCount, VkImageView& imageView);

    bool transitionImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t layerCount);
    bool runEquirectToCube();
    bool generateEnvironmentMipmaps();
    bool runIrradiance();
    bool runPrefilter();
    bool runBrdfLut();

    bool createComputePipeline(const char* shaderPath, VkDescriptorSetLayout descriptorSetLayout, uint32_t pushConstantSize, VkPipelineLayout& pipelineLayout, VkPipeline& pipeline);
    bool dispatchCompute(VkDescriptorSetLayout descriptorSetLayout, VkPipelineLayout pipelineLayout, VkPipeline pipeline, const VkWriteDescriptorSet* writes, uint32_t writeCount, const void* pushConstants, uint32_t pushConstantSize, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;

    VkImage sourceEquirectImage = VK_NULL_HANDLE;
    VkDeviceMemory sourceEquirectMemory = VK_NULL_HANDLE;
    VkImageView sourceEquirectView = VK_NULL_HANDLE;

    VkImage environmentCubeImage = VK_NULL_HANDLE;
    VkDeviceMemory environmentCubeMemory = VK_NULL_HANDLE;
    VkImageView environmentCubeView = VK_NULL_HANDLE;

    VkImage irradianceCubeImage = VK_NULL_HANDLE;
    VkDeviceMemory irradianceCubeMemory = VK_NULL_HANDLE;
    VkImageView irradianceCubeView = VK_NULL_HANDLE;

    VkImage prefilteredCubeImage = VK_NULL_HANDLE;
    VkDeviceMemory prefilteredCubeMemory = VK_NULL_HANDLE;
    VkImageView prefilteredCubeView = VK_NULL_HANDLE;

    VkImage brdfLutImage = VK_NULL_HANDLE;
    VkDeviceMemory brdfLutMemory = VK_NULL_HANDLE;
    VkImageView brdfLutView = VK_NULL_HANDLE;

    VkSampler iblSampler = VK_NULL_HANDLE;
    VkSampler equirectSampler = VK_NULL_HANDLE;
    bool initialized = false;
};
