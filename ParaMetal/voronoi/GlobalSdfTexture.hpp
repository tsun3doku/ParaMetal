#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

namespace voronoi {

class GlobalSdfTexture {
public:
    GlobalSdfTexture(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);
    ~GlobalSdfTexture();

    GlobalSdfTexture(const GlobalSdfTexture&) = delete;
    GlobalSdfTexture& operator=(const GlobalSdfTexture&) = delete;
    GlobalSdfTexture(GlobalSdfTexture&&) noexcept = default;
    GlobalSdfTexture& operator=(GlobalSdfTexture&&) noexcept = default;

    bool upload(
        const std::vector<float>& sdfValues,
        uint32_t channelCount,
        const glm::ivec3& gridDim);

    void cleanup();

    const std::vector<VkImageView>& getSdfImageViews() const { return sdfImageViews; }
    VkSampler getSampler() const { return sdfSampler; }

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;

    std::vector<VkImage> sdfImages;
    std::vector<VkDeviceMemory> sdfImageMemories;
    std::vector<VkImageView> sdfImageViews;

    VkSampler sdfSampler = VK_NULL_HANDLE;
};

} // namespace voronoi
