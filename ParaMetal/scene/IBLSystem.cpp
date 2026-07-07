#include <stb_image.h>

#include "IBLSystem.hpp"

#include "libs/tinyexr/tinyexr.h"
#include "util/file_utils.h"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

IBLSystem::IBLSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool) {
}

IBLSystem::~IBLSystem() {
    cleanup();
}

bool IBLSystem::initialize(const std::string& hdrPath) {
    cleanup();

    if (!loadEquirectangularHDR(hdrPath)) {
        cleanup();
        return false;
    }
    if (!createImages()) {
        cleanup();
        return false;
    }
    if (!createSampler()) {
        cleanup();
        return false;
    }
    if (!createEquirectSampler()) {
        cleanup();
        return false;
    }
    if (!generateIBLMaps()) {
        cleanup();
        return false;
    }

    initialized = true;
    return true;
}

void IBLSystem::cleanup() {
    initialized = false;
    const VkDevice device = vulkanDevice.getDevice();
    if (device == VK_NULL_HANDLE) {
        return;
    }

    if (iblSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, iblSampler, nullptr);
        iblSampler = VK_NULL_HANDLE;
    }

    if (equirectSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, equirectSampler, nullptr);
        equirectSampler = VK_NULL_HANDLE;
    }

    if (brdfLutView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, brdfLutView, nullptr);
        brdfLutView = VK_NULL_HANDLE;
    }
    if (brdfLutImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, brdfLutImage, nullptr);
        brdfLutImage = VK_NULL_HANDLE;
    }
    memoryAllocator.freeImageMemory(brdfLutMemory);
    brdfLutMemory = VK_NULL_HANDLE;

    if (prefilteredCubeView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, prefilteredCubeView, nullptr);
        prefilteredCubeView = VK_NULL_HANDLE;
    }
    if (prefilteredCubeImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, prefilteredCubeImage, nullptr);
        prefilteredCubeImage = VK_NULL_HANDLE;
    }
    memoryAllocator.freeImageMemory(prefilteredCubeMemory);
    prefilteredCubeMemory = VK_NULL_HANDLE;

    if (irradianceCubeView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, irradianceCubeView, nullptr);
        irradianceCubeView = VK_NULL_HANDLE;
    }
    if (irradianceCubeImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, irradianceCubeImage, nullptr);
        irradianceCubeImage = VK_NULL_HANDLE;
    }
    memoryAllocator.freeImageMemory(irradianceCubeMemory);
    irradianceCubeMemory = VK_NULL_HANDLE;

    if (environmentCubeView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, environmentCubeView, nullptr);
        environmentCubeView = VK_NULL_HANDLE;
    }
    if (environmentCubeImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, environmentCubeImage, nullptr);
        environmentCubeImage = VK_NULL_HANDLE;
    }
    memoryAllocator.freeImageMemory(environmentCubeMemory);
    environmentCubeMemory = VK_NULL_HANDLE;

    if (sourceEquirectView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, sourceEquirectView, nullptr);
        sourceEquirectView = VK_NULL_HANDLE;
    }
    if (sourceEquirectImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, sourceEquirectImage, nullptr);
        sourceEquirectImage = VK_NULL_HANDLE;
    }
    memoryAllocator.freeImageMemory(sourceEquirectMemory);
    sourceEquirectMemory = VK_NULL_HANDLE;
}

bool IBLSystem::loadEquirectangularHDR(const std::string& hdrPath) {
    int width = 0;
    int height = 0;
    int channels = 0;
    bool loadedWithTinyExr = false;

    std::string extension;
    const size_t extensionStart = hdrPath.find_last_of('.');
    if (extensionStart != std::string::npos) {
        extension = hdrPath.substr(extensionStart);
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    }

    float* pixels = nullptr;
    if (extension == ".exr") {
        const char* error = nullptr;
        if (LoadEXR(&pixels, &width, &height, hdrPath.c_str(), &error) != TINYEXR_SUCCESS) {
            std::cerr << "[IBLSystem] Failed to load EXR environment: " << hdrPath;
            if (error) {
                std::cerr << " (" << error << ")";
                FreeEXRErrorMessage(error);
            }
            std::cerr << std::endl;
            return false;
        }
        channels = 4;
        loadedWithTinyExr = true;
    } else {
        pixels = stbi_loadf(hdrPath.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    }

    if (!pixels || width <= 0 || height <= 0) {
        if (pixels) {
            if (loadedWithTinyExr) {
                std::free(pixels);
            } else {
                stbi_image_free(pixels);
            }
        }
        std::cerr << "[IBLSystem] Failed to load HDR environment: " << hdrPath << std::endl;
        return false;
    }

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4u * sizeof(float);

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingOffset = 0;
    void* mapped = nullptr;
    if (createStagingBuffer(memoryAllocator, imageSize, stagingBuffer, stagingOffset, &mapped) != VK_SUCCESS || !mapped) {
        if (loadedWithTinyExr) {
            std::free(pixels);
        } else {
            stbi_image_free(pixels);
        }
        std::cerr << "[IBLSystem] Failed to create HDR staging buffer" << std::endl;
        return false;
    }
    std::memcpy(mapped, pixels, static_cast<size_t>(imageSize));
    if (loadedWithTinyExr) {
        std::free(pixels);
    } else {
        stbi_image_free(pixels);
    }

    if (!create2DImage(
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            sourceEquirectImage,
            sourceEquirectMemory)) {
        memoryAllocator.free(stagingBuffer, stagingOffset);
        return false;
    }

    if (!transitionImage(sourceEquirectImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, 1)) {
        memoryAllocator.free(stagingBuffer, stagingOffset);
        return false;
    }

    VkCommandBuffer commandBuffer = commandPool.beginCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        memoryAllocator.free(stagingBuffer, stagingOffset);
        return false;
    }

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = stagingOffset;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, sourceEquirectImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    const bool submitted = commandPool.endCommands(commandBuffer);
    memoryAllocator.free(stagingBuffer, stagingOffset);
    if (!submitted) {
        return false;
    }

    if (!transitionImage(sourceEquirectImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1)) {
        return false;
    }
    if (!create2DView(sourceEquirectImage, VK_FORMAT_R32G32B32A32_SFLOAT, sourceEquirectView)) {
        return false;
    }

    return true;
}

bool IBLSystem::createImages() {
    if (!createCubeImage(EnvironmentCubeSize, EnvironmentMipLevels,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            environmentCubeImage, environmentCubeMemory)) {
        return false;
    }
    if (!createCubeView(environmentCubeImage, VK_FORMAT_R16G16B16A16_SFLOAT, EnvironmentMipLevels, environmentCubeView)) {
        return false;
    }

    if (!createCubeImage(IrradianceCubeSize, 1, VK_FORMAT_R16G16B16A16_SFLOAT, irradianceCubeImage, irradianceCubeMemory)) {
        return false;
    }
    if (!createCubeView(irradianceCubeImage, VK_FORMAT_R16G16B16A16_SFLOAT, 1, irradianceCubeView)) {
        return false;
    }

    if (!createCubeImage(PrefilteredCubeSize, PrefilteredMipLevels, VK_FORMAT_R16G16B16A16_SFLOAT, prefilteredCubeImage, prefilteredCubeMemory)) {
        return false;
    }
    if (!createCubeView(prefilteredCubeImage, VK_FORMAT_R16G16B16A16_SFLOAT, PrefilteredMipLevels, prefilteredCubeView)) {
        return false;
    }

    if (!create2DImage(
            BrdfLutSize,
            BrdfLutSize,
            VK_FORMAT_R16G16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            brdfLutImage,
            brdfLutMemory)) {
        return false;
    }
    if (!create2DView(brdfLutImage, VK_FORMAT_R16G16_SFLOAT, brdfLutView)) {
        return false;
    }

    return true;
}

bool IBLSystem::createSampler() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(vulkanDevice.getPhysicalDevice(), &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(EnvironmentMipLevels - 1);
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &iblSampler) != VK_SUCCESS) {
        std::cerr << "[IBLSystem] Failed to create IBL sampler" << std::endl;
        return false;
    }
    return true;
}

bool IBLSystem::createEquirectSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &equirectSampler) != VK_SUCCESS) {
        std::cerr << "[IBLSystem] Failed to create equirect sampler" << std::endl;
        return false;
    }
    return true;
}

bool IBLSystem::generateIBLMaps() {
    if (!transitionImage(environmentCubeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, EnvironmentMipLevels, 6)) {
        return false;
    }
    if (!transitionImage(irradianceCubeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 6)) {
        return false;
    }
    if (!transitionImage(prefilteredCubeImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, PrefilteredMipLevels, 6)) {
        return false;
    }
    if (!transitionImage(brdfLutImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 1, 1)) {
        return false;
    }

    if (!runEquirectToCube()) return false;
    if (!generateEnvironmentMipmaps()) return false;
    if (!runIrradiance()) return false;
    if (!runPrefilter()) return false;
    if (!runBrdfLut()) return false;

    if (!transitionImage(irradianceCubeImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 6)) {
        return false;
    }
    if (!transitionImage(prefilteredCubeImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, PrefilteredMipLevels, 6)) {
        return false;
    }
    if (!transitionImage(brdfLutImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1, 1)) {
        return false;
    }

    return true;
}

bool IBLSystem::create2DImage(uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(vulkanDevice.getDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "[IBLSystem] Failed to create 2D image" << std::endl;
        return false;
    }
    memory = memoryAllocator.allocateImageMemory(image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    return memory != VK_NULL_HANDLE;
}

bool IBLSystem::createCubeImage(uint32_t size, uint32_t mipLevels, VkFormat format, VkImage& image, VkDeviceMemory& memory) {
    return createCubeImage(size, mipLevels, format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        image, memory);
}

bool IBLSystem::createCubeImage(uint32_t size, uint32_t mipLevels, VkFormat format, VkImageUsageFlags usage, VkImage& image, VkDeviceMemory& memory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {size, size, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 6;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(vulkanDevice.getDevice(), &imageInfo, nullptr, &image) != VK_SUCCESS) {
        std::cerr << "[IBLSystem] Failed to create cube image" << std::endl;
        return false;
    }
    memory = memoryAllocator.allocateImageMemory(image, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    return memory != VK_NULL_HANDLE;
}

bool IBLSystem::create2DView(VkImage image, VkFormat format, VkImageView& imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    return vkCreateImageView(vulkanDevice.getDevice(), &viewInfo, nullptr, &imageView) == VK_SUCCESS;
}

bool IBLSystem::createCubeView(VkImage image, VkFormat format, uint32_t mipLevels, VkImageView& imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;
    return vkCreateImageView(vulkanDevice.getDevice(), &viewInfo, nullptr, &imageView) == VK_SUCCESS;
}

bool IBLSystem::createStorageView(VkImage image, VkImageViewType viewType, VkFormat format, uint32_t mipLevel, uint32_t layerCount, VkImageView& imageView) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = mipLevel;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;
    return vkCreateImageView(vulkanDevice.getDevice(), &viewInfo, nullptr, &imageView) == VK_SUCCESS;
}

bool IBLSystem::transitionImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels, uint32_t layerCount) {
    VkCommandBuffer commandBuffer = commandPool.beginCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = layerCount;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else {
        commandPool.endCommands(commandBuffer);
        std::cerr << "[IBLSystem] Unsupported image layout transition" << std::endl;
        return false;
    }

    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    return commandPool.endCommands(commandBuffer);
}

bool IBLSystem::createComputePipeline(const char* shaderPath, VkDescriptorSetLayout descriptorSetLayout, uint32_t pushConstantSize, VkPipelineLayout& pipelineLayout, VkPipeline& pipeline) {
    auto shaderCode = readFile(shaderPath);
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, shaderCode, shaderModule) != VK_SUCCESS) {
        std::cerr << "[IBLSystem] Failed to create shader module: " << shaderPath << std::endl;
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = pushConstantSize > 0 ? 1u : 0u;
    layoutInfo.pPushConstantRanges = pushConstantSize > 0 ? &pushRange : nullptr;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), shaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = pipelineLayout;

    const VkResult result = vkCreateComputePipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), shaderModule, nullptr);
    return result == VK_SUCCESS;
}

bool IBLSystem::dispatchCompute(VkDescriptorSetLayout descriptorSetLayout, VkPipelineLayout pipelineLayout, VkPipeline pipeline, const VkWriteDescriptorSet* writes, uint32_t writeCount, const void* pushConstants, uint32_t pushConstantSize, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ) {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &descriptorSet) != VK_SUCCESS) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
        return false;
    }

    std::vector<VkWriteDescriptorSet> descriptorWrites(writes, writes + writeCount);
    for (VkWriteDescriptorSet& write : descriptorWrites) {
        write.dstSet = descriptorSet;
    }
    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    VkCommandBuffer commandBuffer = commandPool.beginCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
        return false;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    if (pushConstants && pushConstantSize > 0) {
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstantSize, pushConstants);
    }
    vkCmdDispatch(commandBuffer, groupsX, groupsY, groupsZ);

    const bool submitted = commandPool.endCommands(commandBuffer);
    vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
    return submitted;
}

bool IBLSystem::runEquirectToCube() {
    VkImageView outputView = VK_NULL_HANDLE;
    if (!createStorageView(environmentCubeImage, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 6, outputView)) {
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
        return false;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool ok = createComputePipeline("shaders/ibl_equirect_to_cube_comp.spv", descriptorSetLayout, sizeof(EquirectPC), pipelineLayout, pipeline);
    if (ok) {
        VkDescriptorImageInfo sourceInfo{VK_NULL_HANDLE, sourceEquirectView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo samplerInfo{equirectSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &sourceInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &samplerInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &outputInfo;
        EquirectPC push{EnvironmentCubeSize};
        ok = dispatchCompute(descriptorSetLayout, pipelineLayout, pipeline, writes, 3, &push, sizeof(push), (EnvironmentCubeSize + ComputeLocalSize - 1u) / ComputeLocalSize, (EnvironmentCubeSize + ComputeLocalSize - 1u) / ComputeLocalSize, 6);
    }

    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
    vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
    return ok;
}

bool IBLSystem::generateEnvironmentMipmaps() {
    VkCommandBuffer commandBuffer = commandPool.beginCommands();
    if (commandBuffer == VK_NULL_HANDLE) {
        return false;
    }

    VkImageMemoryBarrier lod0ToSrc{};
    lod0ToSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    lod0ToSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lod0ToSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lod0ToSrc.image = environmentCubeImage;
    lod0ToSrc.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    lod0ToSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    lod0ToSrc.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    lod0ToSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    lod0ToSrc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    lod0ToSrc.subresourceRange.baseMipLevel = 0;
    lod0ToSrc.subresourceRange.levelCount = 1;
    lod0ToSrc.subresourceRange.baseArrayLayer = 0;
    lod0ToSrc.subresourceRange.layerCount = 6;
    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &lod0ToSrc);

    VkImageMemoryBarrier restToDst{};
    restToDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    restToDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    restToDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    restToDst.image = environmentCubeImage;
    restToDst.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    restToDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    restToDst.srcAccessMask = 0;
    restToDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    restToDst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    restToDst.subresourceRange.baseMipLevel = 1;
    restToDst.subresourceRange.levelCount = EnvironmentMipLevels - 1;
    restToDst.subresourceRange.baseArrayLayer = 0;
    restToDst.subresourceRange.layerCount = 6;
    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &restToDst);

    int32_t mipWidth = static_cast<int32_t>(EnvironmentCubeSize);
    int32_t mipHeight = static_cast<int32_t>(EnvironmentCubeSize);
    for (uint32_t mip = 1; mip < EnvironmentMipLevels; ++mip) {
        VkImageMemoryBarrier beforeBlit{};
        beforeBlit.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        beforeBlit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeBlit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        beforeBlit.image = environmentCubeImage;
        beforeBlit.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        beforeBlit.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        beforeBlit.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        beforeBlit.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        beforeBlit.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        beforeBlit.subresourceRange.baseMipLevel = mip - 1;
        beforeBlit.subresourceRange.levelCount = 1;
        beforeBlit.subresourceRange.baseArrayLayer = 0;
        beforeBlit.subresourceRange.layerCount = 6;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr, 0, nullptr, 1, &beforeBlit);

        mipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
        mipHeight = mipHeight > 1 ? mipHeight / 2 : 1;

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = mip - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 6;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {std::max(mipWidth * 2, 1), std::max(mipHeight * 2, 1), 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = mip;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 6;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth, mipHeight, 1};
        vkCmdBlitImage(commandBuffer,
            environmentCubeImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            environmentCubeImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        VkImageMemoryBarrier afterBlit{};
        afterBlit.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        afterBlit.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        afterBlit.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        afterBlit.image = environmentCubeImage;
        afterBlit.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        afterBlit.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        afterBlit.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        afterBlit.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        afterBlit.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        afterBlit.subresourceRange.baseMipLevel = mip;
        afterBlit.subresourceRange.levelCount = 1;
        afterBlit.subresourceRange.baseArrayLayer = 0;
        afterBlit.subresourceRange.layerCount = 6;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, nullptr, 0, nullptr, 1, &afterBlit);
    }

    // Transition the whole image into SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier toReadOnly{};
    toReadOnly.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toReadOnly.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadOnly.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toReadOnly.image = environmentCubeImage;
    toReadOnly.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toReadOnly.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toReadOnly.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    toReadOnly.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    toReadOnly.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toReadOnly.subresourceRange.baseMipLevel = 0;
    toReadOnly.subresourceRange.levelCount = EnvironmentMipLevels;
    toReadOnly.subresourceRange.baseArrayLayer = 0;
    toReadOnly.subresourceRange.layerCount = 6;
    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        0, nullptr, 0, nullptr, 1, &toReadOnly);

    return commandPool.endCommands(commandBuffer);
}

bool IBLSystem::runIrradiance() {
    VkImageView outputView = VK_NULL_HANDLE;
    if (!createStorageView(irradianceCubeImage, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 6, outputView)) {
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
        return false;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool ok = createComputePipeline("shaders/ibl_irradiance_comp.spv", descriptorSetLayout, sizeof(IrradiancePC), pipelineLayout, pipeline);
    if (ok) {
        VkDescriptorImageInfo sourceInfo{VK_NULL_HANDLE, environmentCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo samplerInfo{iblSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &sourceInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &samplerInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &outputInfo;
        IrradiancePC push{IrradianceCubeSize};
        ok = dispatchCompute(descriptorSetLayout, pipelineLayout, pipeline, writes, 3, &push, sizeof(push), (IrradianceCubeSize + ComputeLocalSize - 1u) / ComputeLocalSize, (IrradianceCubeSize + ComputeLocalSize - 1u) / ComputeLocalSize, 6);
    }

    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
    vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
    return ok;
}

bool IBLSystem::runPrefilter() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        return false;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool ok = createComputePipeline("shaders/ibl_prefilter_comp.spv", descriptorSetLayout, sizeof(PrefilterPC), pipelineLayout, pipeline);

    for (uint32_t mip = 0; ok && mip < PrefilteredMipLevels; ++mip) {
        const uint32_t faceSize = std::max(1u, PrefilteredCubeSize >> mip);
        VkImageView outputView = VK_NULL_HANDLE;
        if (!createStorageView(prefilteredCubeImage, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_R16G16B16A16_SFLOAT, mip, 6, outputView)) {
            ok = false;
            break;
        }

        VkDescriptorImageInfo sourceInfo{VK_NULL_HANDLE, environmentCubeView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo samplerInfo{iblSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].descriptorCount = 1;
        writes[0].pImageInfo = &sourceInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &samplerInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &outputInfo;
        PrefilterPC push{faceSize, mip, PrefilteredMipLevels, EnvironmentCubeSize};
        ok = dispatchCompute(descriptorSetLayout, pipelineLayout, pipeline, writes, 3, &push, sizeof(push), (faceSize + ComputeLocalSize - 1u) / ComputeLocalSize, (faceSize + ComputeLocalSize - 1u) / ComputeLocalSize, 6);
        vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
    }

    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
    return ok;
}

bool IBLSystem::runBrdfLut() {
    VkImageView outputView = VK_NULL_HANDLE;
    if (!createStorageView(brdfLutImage, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R16G16_SFLOAT, 0, 1, outputView)) {
        return false;
    }

    VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
        return false;
    }

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool ok = createComputePipeline("shaders/ibl_brdf_lut_comp.spv", descriptorSetLayout, sizeof(BrdfPC), pipelineLayout, pipeline);
    if (ok) {
        VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, outputView, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.descriptorCount = 1;
        write.pImageInfo = &outputInfo;
        BrdfPC push{BrdfLutSize};
        ok = dispatchCompute(descriptorSetLayout, 
            pipelineLayout, 
            pipeline, 
            &write, 
            1, 
            &push, 
            sizeof(push), 
            (BrdfLutSize + ComputeLocalSize - 1u) / ComputeLocalSize, 
            (BrdfLutSize + ComputeLocalSize - 1u) / ComputeLocalSize, 1);
    }

    if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
    if (pipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
    vkDestroyImageView(vulkanDevice.getDevice(), outputView, nullptr);
    return ok;
}
