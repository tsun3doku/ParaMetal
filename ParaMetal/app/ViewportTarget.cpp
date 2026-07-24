#include "ViewportTarget.hpp"

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

void ViewportTarget::initialize(VulkanDevice& device) {
    cleanup();
    vulkanDevice = &device;
}

bool ViewportTarget::update(VkImage updatedImage, VkFormat updatedFormat, VkExtent2D updatedExtent) {
    if (!vulkanDevice ||
        vulkanDevice->getDevice() == VK_NULL_HANDLE ||
        updatedImage == VK_NULL_HANDLE ||
        updatedFormat == VK_FORMAT_UNDEFINED ||
        updatedExtent.width == 0 ||
        updatedExtent.height == 0) {
        return false;
    }

    if (image == updatedImage &&
        imageFormat == updatedFormat &&
        extent.width == updatedExtent.width &&
        extent.height == updatedExtent.height) {
        return true;
    }

    cleanup();

    VkImageView imageView = VK_NULL_HANDLE;
    if (createImageView(
            *vulkanDevice,
            updatedImage,
            updatedFormat,
            VK_IMAGE_ASPECT_COLOR_BIT,
            imageView) != VK_SUCCESS) {
        return false;
    }

    image = updatedImage;
    imageFormat = updatedFormat;
    extent = updatedExtent;
    imageViews.push_back(imageView);
    return true;
}

void ViewportTarget::cleanup() {
    if (vulkanDevice && vulkanDevice->getDevice() != VK_NULL_HANDLE) {
        for (VkImageView imageView : imageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(vulkanDevice->getDevice(), imageView, nullptr);
            }
        }
    }

    imageViews.clear();
    image = VK_NULL_HANDLE;
    imageFormat = VK_FORMAT_UNDEFINED;
    extent = {};
}
