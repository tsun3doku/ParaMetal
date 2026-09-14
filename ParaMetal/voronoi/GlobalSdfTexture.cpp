#include "voronoi/GlobalSdfTexture.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanImage.hpp"

#include <iostream>

namespace voronoi {

GlobalSdfTexture::GlobalSdfTexture(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool) {
}

GlobalSdfTexture::~GlobalSdfTexture() {
    cleanup();
}

void GlobalSdfTexture::cleanup() {
    if (sdfSampler != VK_NULL_HANDLE) {
        vkDestroySampler(vulkanDevice.getDevice(), sdfSampler, nullptr);
        sdfSampler = VK_NULL_HANDLE;
    }

    for (VkImageView view : sdfImageViews) {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(vulkanDevice.getDevice(), view, nullptr);
    }
    sdfImageViews.clear();

    for (size_t i = 0; i < sdfImages.size(); ++i) {
        if (sdfImages[i] != VK_NULL_HANDLE) vkDestroyImage(vulkanDevice.getDevice(), sdfImages[i], nullptr);
        if (i < sdfImageMemories.size() && sdfImageMemories[i] != VK_NULL_HANDLE) {
            memoryAllocator.freeImageMemory(sdfImageMemories[i]);
        }
    }
    sdfImages.clear();
    sdfImageMemories.clear();
}

bool GlobalSdfTexture::upload(
    const std::vector<float>& sdfValues,
    uint32_t channelCount,
    const glm::ivec3& gridDim) {
    cleanup();

    const size_t voxelsPerChannel = static_cast<size_t>(gridDim.x) * gridDim.y * gridDim.z;
    if (channelCount == 0 || voxelsPerChannel == 0 || sdfValues.size() < voxelsPerChannel * channelCount) {
        return false;
    }

    // 1. Create linear sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &sdfSampler) != VK_SUCCESS) {
        std::cerr << "[GlobalSdfTexture] Failed to create 3D SDF sampler." << std::endl;
        return false;
    }

    // 2. Upload SDF textures
    for (uint32_t ch = 0; ch < channelCount; ++ch) {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        if (uploadR32FloatImage3D(vulkanDevice, memoryAllocator, commandPool,
                sdfValues.data() + ch * voxelsPerChannel, gridDim, image, mem, view) != VK_SUCCESS) {
            std::cerr << "[GlobalSdfTexture] Failed to upload 3D SDF channel " << ch << std::endl;
            cleanup();
            return false;
        }
        sdfImages.push_back(image);
        sdfImageMemories.push_back(mem);
        sdfImageViews.push_back(view);
    }

    return true;
}

} // namespace voronoi
