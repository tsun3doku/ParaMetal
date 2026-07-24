#pragma once

#include <vulkan/vulkan.h>

#include <vector>

class VulkanDevice;

class ViewportTarget {
public:
    void initialize(VulkanDevice& vulkanDevice);
    bool update(VkImage image, VkFormat format, VkExtent2D extent);
    void cleanup();

    const std::vector<VkImageView>& getImageViews() const {
        return imageViews;
    }

    VkImage getImage() const {
        return image;
    }

    VkFormat getImageFormat() const {
        return imageFormat;
    }

    VkExtent2D getExtent() const {
        return extent;
    }

private:
    VulkanDevice* vulkanDevice = nullptr;
    VkImage image = VK_NULL_HANDLE;
    std::vector<VkImageView> imageViews;
    VkFormat imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
};
