#include "IntrinsicRenderer.hpp"

#include "mesh/remesher/SupportingHalfedge.hpp"
#include "mesh/remesher/iODT.hpp"
#include "scene/ModelSelection.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "util/file_utils.h"
#include "render/RenderConfig.hpp"
#include "util/Structs.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <glm/glm.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <unordered_set>

static_assert(sizeof(IntrinsicVertexData) == 32);
static_assert(offsetof(IntrinsicVertexData, intrinsicVertexId) == 12);

IntrinsicRenderer::IntrinsicRenderer(VulkanDevice& device, MemoryAllocator& allocator, UniformBufferManager& uboManager, CommandPool& commandPool)
    : vulkanDevice(device), allocator(allocator), uniformBufferManager(uboManager), renderCommandPool(commandPool) {
}

void IntrinsicRenderer::setModelRegistry(ModelRegistry* registry) {
    modelRegistry = registry;
}

glm::mat4 IntrinsicRenderer::resolveModelMatrix(
    const RemeshDisplayController::Config& config) const {
    glm::mat4 matrix{1.0f};
    if (modelRegistry && config.runtimeModelId != 0 &&
        modelRegistry->tryGetModelMatrix(config.runtimeModelId, matrix)) {
        return matrix;
    }
    return config.modelMatrix;
}

float IntrinsicRenderer::resolveVertexMarkerRadius(
    const RemeshDisplayController::Config& config) const {
    if (config.averageTriangleArea > 0.0f) {
        return std::max(
            1e-5f, std::sqrt(config.averageTriangleArea) * 0.12f);
    }
    return VertexHighlightRadiusWorld;
}

IntrinsicRenderer::~IntrinsicRenderer() {
    cleanup();
}

bool IntrinsicRenderer::initializeGeometry(VkRenderPass renderPass, uint32_t updatedMaxFramesInFlight, uint32_t subpassIndex) {
    if (surfaceInitialized) {
        return true;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    const uint32_t zeroMask = 0;
    if (createTexelBuffer(
            allocator,
            vulkanDevice,
            &zeroMask,
            sizeof(zeroMask),
            VK_FORMAT_R32_UINT,
            zeroElementMaskBuffer,
            zeroElementMaskOffset,
            zeroElementMaskView) != VK_SUCCESS ||
        !createWireframeTexture() ||
        !createSupportingHalfedgeDescriptorPool(maxFramesInFlight) ||
        !createSupportingHalfedgeDescriptorSetLayout() ||
        !createSupportingHalfedgePipeline(renderPass, subpassIndex)) {
        cleanup();
        return false;
    }

    surfaceInitialized = true;
    return true;
}

bool IntrinsicRenderer::initializeOverlay(VkRenderPass renderPass, uint32_t updatedMaxFramesInFlight, uint32_t subpassIndex) {
    if (overlayInitialized) {
        return true;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    if (!createWireframeTexture() ||
        !createNormalDescriptorPool(maxFramesInFlight) ||
        !createNormalDescriptorSetLayout() ||
        !createNormalPipelineLayout() ||
        !createNormalPipeline(
            renderPass, subpassIndex,
            "shaders/intrinsic_normals_vert.spv",
            "shaders/intrinsic_normals_geom.spv",
            "shaders/intrinsic_normals_frag.spv",
            intrinsicNormalsPipeline) ||
        !createNormalPipeline(
            renderPass, subpassIndex,
            "shaders/intrinsic_vertex_normals_vert.spv",
            "shaders/intrinsic_vertex_normals_geom.spv",
            "shaders/intrinsic_vertex_normals_frag.spv",
            intrinsicVertexNormalsPipeline) ||
        !createXrayWireframePipeline(renderPass, subpassIndex)) {
        cleanupOverlay();
        return false;
    }

    overlayInitialized = true;
    return true;
}

uint32_t IntrinsicRenderer::calculateMipLevels(uint32_t width, uint32_t height) {
    uint32_t levels = 1;
    while (width > 1 || height > 1) {
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);
        ++levels;
    }
    return levels;
}

bool IntrinsicRenderer::createWireframeTexture() {
    if (wireframeTextureSampler != VK_NULL_HANDLE ||
        wireframeTextureView != VK_NULL_HANDLE ||
        wireframeTextureImage != VK_NULL_HANDLE ||
        wireframeTextureMemory != VK_NULL_HANDLE) {
        return true;
    }

    const uint32_t width = 4096;
    const uint32_t height = 1;
    const float thickness = 1.5f;
    const VkFormat wireFormat = VK_FORMAT_R8G8B8A8_UNORM;
    const uint32_t mipLevels = calculateMipLevels(width, height);

    std::vector<uint8_t> pixels;
    std::vector<VkBufferImageCopy> copyRegions;
    pixels.reserve(width * height * 4 * mipLevels);
    copyRegions.reserve(mipLevels);

    uint32_t mipWidth = width;
    uint32_t mipHeight = height;
    for (uint32_t level = 0; level < mipLevels; ++level) {
        const size_t levelOffset = pixels.size();
        const size_t levelSize = static_cast<size_t>(mipWidth) * static_cast<size_t>(mipHeight) * 4;
        pixels.resize(levelOffset + levelSize, 0);

        float mipWidthFloat = static_cast<float>(mipWidth);
        float levelThickness;
        if (thickness < mipWidthFloat / 3.0f) {
            levelThickness = thickness;
        } else {
            levelThickness = mipWidthFloat / 3.0f;
        }
        int integerThickness = static_cast<int>(std::floor(levelThickness / 2.0f));
        float fractionalThickness = std::fmod(levelThickness, 2.0f);

        for (uint32_t i = 0; i < mipWidth; ++i) {
            const size_t idx = levelOffset + static_cast<size_t>(i) * 4;
            pixels[idx + 0] = 0;
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
            pixels[idx + 3] = 0;

            if (i < static_cast<uint32_t>(integerThickness)) {
                pixels[idx + 3] = 255;
            } else if (i == static_cast<uint32_t>(integerThickness) && fractionalThickness > 0.0f) {
                pixels[idx + 3] = static_cast<uint8_t>(fractionalThickness * 255.0f);
            } else if (i >= mipWidth - static_cast<uint32_t>(integerThickness)) {
                pixels[idx + 3] = 255;
            } else if (i == mipWidth - 1 - static_cast<uint32_t>(integerThickness) && fractionalThickness > 0.0f) {
                pixels[idx + 3] = static_cast<uint8_t>(fractionalThickness * 255.0f);
            }
        }

        VkBufferImageCopy region{};
        region.bufferOffset = static_cast<VkDeviceSize>(levelOffset);
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = level;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {mipWidth, mipHeight, 1};
        copyRegions.push_back(region);

        if (mipWidth > 1) {
            mipWidth /= 2;
        }
        if (mipHeight > 1) {
            mipHeight /= 2;
        }
    }

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize stagingBufferOffset = 0;
    void* data = nullptr;
    if (createStagingBuffer(allocator, imageSize, stagingBuffer, stagingBufferOffset, &data) != VK_SUCCESS || !data) {
        std::cerr << "[IntrinsicRenderer] Failed to create wireframe staging buffer" << std::endl;
        return false;
    }
    std::memcpy(data, pixels.data(), static_cast<size_t>(imageSize));

    if (createImage(
            vulkanDevice,
            allocator,
            width,
            height,
            wireFormat,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            wireframeTextureImage,
            wireframeTextureMemory,
            VK_SAMPLE_COUNT_1_BIT,
            mipLevels) != VK_SUCCESS) {
        allocator.free(stagingBuffer, stagingBufferOffset);
        return false;
    }

    if (transitionImageLayout(
            renderCommandPool,
            wireframeTextureImage,
            wireFormat,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            mipLevels) != VK_SUCCESS) {
        allocator.free(stagingBuffer, stagingBufferOffset);
        return false;
    }

    {
        VkCommandBuffer commandBuffer = renderCommandPool.beginCommands();
        if (commandBuffer == VK_NULL_HANDLE) {
            allocator.free(stagingBuffer, stagingBufferOffset);
            return false;
        }
        vkCmdCopyBufferToImage(
            commandBuffer,
            stagingBuffer,
            wireframeTextureImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(copyRegions.size()),
            copyRegions.data());
        renderCommandPool.endCommands(commandBuffer);
    }

    if (transitionImageLayout(
            renderCommandPool,
            wireframeTextureImage,
            wireFormat,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            mipLevels) != VK_SUCCESS) {
        allocator.free(stagingBuffer, stagingBufferOffset);
        return false;
    }

    allocator.free(stagingBuffer, stagingBufferOffset);

    wireframeTextureView = createImageView(
        vulkanDevice,
        wireframeTextureImage,
        wireFormat,
        VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevels);
    if (wireframeTextureView == VK_NULL_HANDLE) {
        return false;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels - 1);
    samplerInfo.anisotropyEnable = VK_FALSE;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &wireframeTextureSampler) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create wireframe sampler" << std::endl;
        return false;
    }

    return true;
}

bool IntrinsicRenderer::createSupportingHalfedgeDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t maxModels = renderconfig::MaxIntrinsicSurfaces;

    std::array<VkDescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxFramesInFlight * maxModels * 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    poolSizes[1].descriptorCount = maxFramesInFlight * maxModels * 12;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = maxFramesInFlight * maxModels;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[3].descriptorCount = maxFramesInFlight * maxModels;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxFramesInFlight * maxModels;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &supportingHalfedgeDescriptorPool) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge descriptor pool" << std::endl;
        return false;
    }
    return true;
}

bool IntrinsicRenderer::createSupportingHalfedgeDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 16> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding = 5;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[6].binding = 6;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[7].binding = 7;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[7].descriptorCount = 1;
    bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[8].binding = 8;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[9].binding = 9;
    bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[10].binding = 10;
    bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[10].descriptorCount = 1;
    bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[13].binding = 13;
    bindings[13].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    bindings[13].descriptorCount = 1;
    bindings[13].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[14].binding = 14;
    bindings[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[14].descriptorCount = 1;
    bindings[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[15].binding = 15;
    bindings[15].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[15].descriptorCount = 1;
    bindings[15].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &supportingHalfedgeDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge descriptor set layout" << std::endl;
        return false;
    }
    return true;
}

bool IntrinsicRenderer::allocateSupportingHalfedgeDescriptorSets(uint64_t surfaceIdentity, uint32_t maxFramesInFlight) {
    if (surfaceIdentity == 0) {
        return false;
    }

    if (supportingHalfedgeDescriptorSetsBySocket.find(surfaceIdentity) != supportingHalfedgeDescriptorSetsBySocket.end()) {
        return true;
    }
    if (supportingHalfedgeDescriptorPool == VK_NULL_HANDLE ||
        supportingHalfedgeDescriptorSetLayout == VK_NULL_HANDLE) {
        return false;
    }

    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, supportingHalfedgeDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = supportingHalfedgeDescriptorPool;
    allocInfo.descriptorSetCount = maxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();

    std::vector<VkDescriptorSet> descriptorSets(maxFramesInFlight);
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to allocate supporting halfedge descriptor sets" << std::endl;
        return false;
    }

    supportingHalfedgeDescriptorSetsBySocket[surfaceIdentity] = std::move(descriptorSets);
    return true;
}

bool IntrinsicRenderer::createNormalDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t maxSurfaces = renderconfig::MaxIntrinsicSurfaces;
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxFramesInFlight * maxSurfaces;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = maxFramesInFlight * maxSurfaces;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxFramesInFlight * maxSurfaces;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &normalDescriptorPool) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create normal descriptor pool" << std::endl;
        return false;
    }
    return true;
}

bool IntrinsicRenderer::createNormalDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &normalDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create normal descriptor set layout" << std::endl;
        return false;
    }
    return true;
}

bool IntrinsicRenderer::allocateNormalDescriptorSets(
    uint64_t socketKey,
    uint32_t maxFramesInFlight,
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets) {
    if (socketKey == 0 || normalDescriptorPool == VK_NULL_HANDLE ||
        normalDescriptorSetLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (sets.find(socketKey) != sets.end()) {
        return true;
    }
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, normalDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = normalDescriptorPool;
    allocInfo.descriptorSetCount = maxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> descriptorSets(maxFramesInFlight);
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to allocate normal descriptor sets" << std::endl;
        return false;
    }
    sets[socketKey] = std::move(descriptorSets);
    return true;
}

bool IntrinsicRenderer::createNormalPipelineLayout() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(NormalPushConstant);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &normalDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &normalPipelineLayout) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create normal pipeline layout" << std::endl;
        return false;
    }
    return true;
}

bool IntrinsicRenderer::updateNormalDescriptorSet(
    uint64_t socketKey,
    const RemeshDisplayController::Config& config,
    uint32_t currentFrame,
    VkBuffer buffer,
    VkDeviceSize bufferOffset,
    VkDeviceSize bufferRange,
    std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets) {
    if (socketKey == 0 || !config.isValid() || buffer == VK_NULL_HANDLE ||
        currentFrame >= maxFramesInFlight) {
        return false;
    }
    const auto it = sets.find(socketKey);
    if (it == sets.end() || currentFrame >= it->second.size()) {
        return false;
    }

    std::array<VkWriteDescriptorSet, 2> writes{};
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBufferManager.getUniformBuffers()[currentFrame];
    uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[currentFrame];
    uboInfo.range = sizeof(UniformBufferObject);

    VkDescriptorBufferInfo dataInfo{};
    dataInfo.buffer = buffer;
    dataInfo.offset = bufferOffset;
    dataInfo.range = bufferRange;

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = it->second[currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &dataInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = it->second[currentFrame];
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &uboInfo;

    vkUpdateDescriptorSets(
        vulkanDevice.getDevice(),
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

void IntrinsicRenderer::cleanupOverlay() {
    if (vulkanDevice.getDevice() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
    }
    if (intrinsicNormalsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), intrinsicNormalsPipeline, nullptr);
        intrinsicNormalsPipeline = VK_NULL_HANDLE;
    }
    if (intrinsicVertexNormalsPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), intrinsicVertexNormalsPipeline, nullptr);
        intrinsicVertexNormalsPipeline = VK_NULL_HANDLE;
    }
    if (xrayWireframePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), xrayWireframePipeline, nullptr);
        xrayWireframePipeline = VK_NULL_HANDLE;
    }
    if (normalPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), normalPipelineLayout, nullptr);
        normalPipelineLayout = VK_NULL_HANDLE;
    }
    if (normalDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), normalDescriptorSetLayout, nullptr);
        normalDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (normalDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), normalDescriptorPool, nullptr);
        normalDescriptorPool = VK_NULL_HANDLE;
    }
    intrinsicNormalsDescriptorSetsBySocket.clear();
    intrinsicVertexNormalsDescriptorSetsBySocket.clear();
    overlayInitialized = false;
}

bool IntrinsicRenderer::updateSupportingHalfedgeDescriptorSet(uint64_t surfaceIdentity, const RemeshDisplayController::Config& config, uint32_t currentFrame) {
    if (surfaceIdentity == 0 || !config.isValid() || currentFrame >= maxFramesInFlight) {
        return false;
    }

    auto descriptorSetIt = supportingHalfedgeDescriptorSetsBySocket.find(surfaceIdentity);
    if (descriptorSetIt == supportingHalfedgeDescriptorSetsBySocket.end()) {
        return false;
    }

    const auto& descriptorSets = descriptorSetIt->second;
    if (currentFrame >= descriptorSets.size()) {
        return false;
    }

    std::array<VkWriteDescriptorSet, 16> descriptorWrites{};

    VkDescriptorBufferInfo uboBufferInfo{};
    uboBufferInfo.buffer = uniformBufferManager.getUniformBuffers()[currentFrame];
    uboBufferInfo.offset = uniformBufferManager.getUniformBufferOffsets()[currentFrame];
    uboBufferInfo.range = sizeof(UniformBufferObject);

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSets[currentFrame];
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uboBufferInfo;

    VkBufferView bufferViews[10] = {
        config.supportingHalfedgeView,
        config.supportingAngleView,
        config.halfedgeView,
        config.edgeView,
        config.triangleView,
        config.lengthView,
        config.inputHalfedgeView,
        config.inputEdgeView,
        config.inputTriangleView,
        config.inputLengthView
    };

    for (int j = 0; j < 10; ++j) {
        descriptorWrites[j + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[j + 1].dstSet = descriptorSets[currentFrame];
        descriptorWrites[j + 1].dstBinding = 1 + j;
        descriptorWrites[j + 1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        descriptorWrites[j + 1].descriptorCount = 1;
        descriptorWrites[j + 1].pTexelBufferView = &bufferViews[j];
    }

    VkDescriptorImageInfo wireframeInfo{};
    wireframeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    wireframeInfo.imageView = wireframeTextureView;
    wireframeInfo.sampler = wireframeTextureSampler;

    descriptorWrites[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[11].dstSet = descriptorSets[currentFrame];
    descriptorWrites[11].dstBinding = 11;
    descriptorWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorWrites[11].descriptorCount = 1;
    descriptorWrites[11].pImageInfo = &wireframeInfo;

    VkBufferView elementViews[2] = {
        config.faceIdView,
        elementMaskView(surfaceIdentity, currentFrame)
    };
    for (int j = 0; j < 2; ++j) {
        descriptorWrites[12 + j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[12 + j].dstSet = descriptorSets[currentFrame];
        descriptorWrites[12 + j].dstBinding = 12 + j;
        descriptorWrites[12 + j].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        descriptorWrites[12 + j].descriptorCount = 1;
        descriptorWrites[12 + j].pTexelBufferView = &elementViews[j];
    }

    VkDescriptorBufferInfo vertexInfo{};
    vertexInfo.buffer = config.intrinsicVertexBuffer;
    vertexInfo.offset = config.intrinsicVertexBufferOffset;
    vertexInfo.range = config.intrinsicVertexCount * sizeof(IntrinsicVertexData);
    descriptorWrites[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[14].dstSet = descriptorSets[currentFrame];
    descriptorWrites[14].dstBinding = 14;
    descriptorWrites[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[14].descriptorCount = 1;
    descriptorWrites[14].pBufferInfo = &vertexInfo;

    VkDescriptorBufferInfo materialInfo{};
    materialInfo.buffer = uniformBufferManager.getMaterialBuffers()[currentFrame];
    materialInfo.offset = uniformBufferManager.getMaterialBufferOffsets()[currentFrame];
    materialInfo.range = sizeof(MaterialUniformBufferObject);
    descriptorWrites[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[15].dstSet = descriptorSets[currentFrame];
    descriptorWrites[15].dstBinding = 15;
    descriptorWrites[15].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[15].descriptorCount = 1;
    descriptorWrites[15].pBufferInfo = &materialInfo;

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    return true;
}

bool IntrinsicRenderer::apply(
    uint64_t socketKey,
    const RemeshDisplayController::Config& config) {
    if (socketKey == 0) {
        return false;
    }
    if (!config.isValid()) {
        remove(socketKey);
        return false;
    }
    if (!surfaceInitialized ||
        supportingHalfedgeDescriptorPool == VK_NULL_HANDLE ||
        supportingHalfedgeDescriptorSetLayout == VK_NULL_HANDLE ||
        supportingHalfedgePipelineLayout == VK_NULL_HANDLE) {
        return false;
    }

    const auto freeSets = [this](
        std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets,
        VkDescriptorPool pool,
        uint64_t key) {
        const auto it = sets.find(key);
        if (it != sets.end() && pool != VK_NULL_HANDLE && !it->second.empty()) {
            vkFreeDescriptorSets(
                vulkanDevice.getDevice(), pool,
                static_cast<uint32_t>(it->second.size()),
                it->second.data());
        }
        sets.erase(key);
    };

    const bool hadSupporting =
        supportingHalfedgeDescriptorSetsBySocket.find(socketKey) !=
        supportingHalfedgeDescriptorSetsBySocket.end();
    if (!hadSupporting &&
        !allocateSupportingHalfedgeDescriptorSets(
            socketKey, maxFramesInFlight)) {
        return false;
    }

    const bool hadFace =
        intrinsicNormalsDescriptorSetsBySocket.find(socketKey) !=
        intrinsicNormalsDescriptorSetsBySocket.end();
    if (config.showFaceNormals && !hadFace) {
        if (normalDescriptorPool == VK_NULL_HANDLE ||
            normalDescriptorSetLayout == VK_NULL_HANDLE ||
            !allocateNormalDescriptorSets(
                socketKey, maxFramesInFlight,
                intrinsicNormalsDescriptorSetsBySocket)) {
            if (!hadSupporting) {
                freeSets(
                    supportingHalfedgeDescriptorSetsBySocket,
                    supportingHalfedgeDescriptorPool,
                    socketKey);
            }
            return false;
        }
    }

    const bool hadVertex =
        intrinsicVertexNormalsDescriptorSetsBySocket.find(socketKey) !=
        intrinsicVertexNormalsDescriptorSetsBySocket.end();
    if (config.showVertexNormals && !hadVertex) {
        if (normalDescriptorPool == VK_NULL_HANDLE ||
            normalDescriptorSetLayout == VK_NULL_HANDLE ||
            !allocateNormalDescriptorSets(
                socketKey, maxFramesInFlight,
                intrinsicVertexNormalsDescriptorSetsBySocket)) {
            if (!hadFace) {
                freeSets(
                    intrinsicNormalsDescriptorSetsBySocket,
                    normalDescriptorPool,
                    socketKey);
            }
            if (!hadSupporting) {
                freeSets(
                    supportingHalfedgeDescriptorSetsBySocket,
                    supportingHalfedgeDescriptorPool,
                    socketKey);
            }
            return false;
        }
    }

    remeshConfigsBySocket[socketKey] = config;
    return true;
}

void IntrinsicRenderer::remove(uint64_t socketKey) {
    releaseDescriptorSetsForSocket(socketKey);
    removeSelection(socketKey);
}

void IntrinsicRenderer::applySelection(
    uint64_t socketKey,
    const RemeshDisplayController::SelectionConfig& config) {
    if (socketKey == 0 || config.surfaceIdentity == 0) {
        return;
    }
    if (config.elementMask.empty()) {
        removeSelection(socketKey);
        return;
    }

    const auto previous = selectionConfigsBySocket.find(socketKey);
    if (previous != selectionConfigsBySocket.end() &&
        (previous->second.surfaceIdentity != config.surfaceIdentity ||
         previous->second.elementType != config.elementType)) {
        const auto buffers = elementMaskBuffersBySocket.find(socketKey);
        if (buffers != elementMaskBuffersBySocket.end()) {
            for (ElementMaskBuffer& frame : buffers->second) {
                frame.revision = UINT64_MAX;
            }
        }
    }

    selectionConfigsBySocket[socketKey] = config;
}

void IntrinsicRenderer::removeSelection(uint64_t socketKey) {
    selectionConfigsBySocket.erase(socketKey);
}

void IntrinsicRenderer::releaseDescriptorSetsForSocket(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    vkDeviceWaitIdle(vulkanDevice.getDevice());

    auto releaseSets = [this, socketKey](
        std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& sets,
        VkDescriptorPool pool) {
        const auto it = sets.find(socketKey);
        if (it != sets.end() && pool != VK_NULL_HANDLE && !it->second.empty()) {
            vkFreeDescriptorSets(
                vulkanDevice.getDevice(), pool,
                static_cast<uint32_t>(it->second.size()),
                it->second.data());
        }
        sets.erase(socketKey);
    };
    releaseSets(
        supportingHalfedgeDescriptorSetsBySocket,
        supportingHalfedgeDescriptorPool);
    releaseSets(
        intrinsicNormalsDescriptorSetsBySocket,
        normalDescriptorPool);
    releaseSets(
        intrinsicVertexNormalsDescriptorSetsBySocket,
        normalDescriptorPool);

    const auto maskIt = elementMaskBuffersBySocket.find(socketKey);
    if (maskIt != elementMaskBuffersBySocket.end()) {
        for (ElementMaskBuffer& buffer : maskIt->second) {
            if (buffer.view != VK_NULL_HANDLE) {
                vkDestroyBufferView(vulkanDevice.getDevice(), buffer.view, nullptr);
            }
            if (buffer.buffer != VK_NULL_HANDLE) {
                allocator.free(buffer.buffer, buffer.offset);
            }
        }
        elementMaskBuffersBySocket.erase(maskIt);
    }

    remeshConfigsBySocket.erase(socketKey);
}

void IntrinsicRenderer::prepareFrame(uint32_t currentFrame, glm::vec2 updatedViewportSize) {
    viewportSize = updatedViewportSize;
    if (currentFrame >= maxFramesInFlight) {
        return;
    }

    if (surfaceInitialized) {
        for (const auto& [surfaceIdentity, config] : remeshConfigsBySocket) {
            if (!config.isValid()) {
                continue;
            }

            const auto selectionIt = selectionConfigsBySocket.find(surfaceIdentity);
            if (selectionIt != selectionConfigsBySocket.end() &&
                !selectionIt->second.elementMask.empty() &&
                ensureElementMaskBuffers(
                    surfaceIdentity,
                    selectionIt->second.elementMask.size(),
                    currentFrame)) {
                ElementMaskBuffer& maskBuffer =
                    elementMaskBuffersBySocket[surfaceIdentity][currentFrame];
                if (maskBuffer.revision != selectionIt->second.revision) {
                    maskBuffer.data = selectionIt->second.elementMask;
                    void* mapped = allocator.getMappedPointer(
                        maskBuffer.buffer, maskBuffer.offset);
                    if (mapped) {
                        std::memcpy(
                            mapped,
                            maskBuffer.data.data(),
                            maskBuffer.data.size() * sizeof(uint32_t));
                        maskBuffer.revision = selectionIt->second.revision;
                    }
                }
            }

            updateSupportingHalfedgeDescriptorSet(
                surfaceIdentity, config, currentFrame);
        }
    }

    if (overlayInitialized) {
        for (const auto& [surfaceIdentity, config] : remeshConfigsBySocket) {
            if (!config.isValid()) {
                continue;
            }
            if (config.showFaceNormals) {
                updateNormalDescriptorSet(
                    surfaceIdentity,
                    config,
                    currentFrame,
                    config.intrinsicTriangleBuffer,
                    config.intrinsicTriangleBufferOffset,
                    config.intrinsicTriangleCount * sizeof(IntrinsicTriangleData),
                    intrinsicNormalsDescriptorSetsBySocket);
            }
            if (config.showVertexNormals) {
                updateNormalDescriptorSet(
                    surfaceIdentity,
                    config,
                    currentFrame,
                    config.intrinsicVertexBuffer,
                    config.intrinsicVertexBufferOffset,
                    config.intrinsicVertexCount * sizeof(IntrinsicVertexData),
                    intrinsicVertexNormalsDescriptorSetsBySocket);
            }
        }
    }
}

bool IntrinsicRenderer::ensureElementMaskBuffers(
    uint64_t surfaceIdentity,
    size_t elementCount,
    uint32_t currentFrame) {
    if (surfaceIdentity == 0 || elementCount == 0 ||
        currentFrame >= maxFramesInFlight) {
        return false;
    }
    if (elementCount >
        vulkanDevice.getPhysicalDeviceProperties().limits.maxTexelBufferElements) {
        std::cerr << "[IntrinsicRenderer] Selection mask exceeds maxTexelBufferElements"
                  << std::endl;
        return false;
    }

    auto& buffers = elementMaskBuffersBySocket[surfaceIdentity];
    if (!buffers.empty() && buffers.front().data.size() != elementCount) {
        if (vkDeviceWaitIdle(vulkanDevice.getDevice()) != VK_SUCCESS) {
            return false;
        }
        for (ElementMaskBuffer& buffer : buffers) {
            if (buffer.view != VK_NULL_HANDLE) {
                vkDestroyBufferView(vulkanDevice.getDevice(), buffer.view, nullptr);
                buffer.view = VK_NULL_HANDLE;
            }
            if (buffer.buffer != VK_NULL_HANDLE) {
                allocator.free(buffer.buffer, buffer.offset);
                buffer.buffer = VK_NULL_HANDLE;
                buffer.offset = 0;
            }
        }
        buffers.clear();
    }

    if (buffers.empty()) {
        buffers.resize(maxFramesInFlight);
    }

    if (buffers[currentFrame].buffer == VK_NULL_HANDLE) {
        ElementMaskBuffer& buffer = buffers[currentFrame];
        buffer.data.assign(elementCount, 0u);
        if (createTexelBuffer(
                allocator,
                vulkanDevice,
                buffer.data.data(),
                buffer.data.size() * sizeof(uint32_t),
                VK_FORMAT_R32_UINT,
                buffer.buffer,
                buffer.offset,
                buffer.view) != VK_SUCCESS) {
            std::cerr << "[IntrinsicRenderer] Failed to create element selection mask buffer" << std::endl;
            return false;
        }
    }

    return true;
}

VkBufferView IntrinsicRenderer::elementMaskView(
    uint64_t surfaceIdentity,
    uint32_t currentFrame) const {
    const auto selectionIt = selectionConfigsBySocket.find(surfaceIdentity);
    if (selectionIt == selectionConfigsBySocket.end() ||
        selectionIt->second.elementMask.empty() ||
        currentFrame >= maxFramesInFlight) {
        return zeroElementMaskView;
    }
    const auto it = elementMaskBuffersBySocket.find(surfaceIdentity);
    if (it == elementMaskBuffersBySocket.end() ||
        currentFrame >= it->second.size()) {
        return zeroElementMaskView;
    }
    return it->second[currentFrame].view;
}

bool IntrinsicRenderer::createSupportingHalfedgePipeline(VkRenderPass renderPass, uint32_t subpassIndex) {
    auto vertShaderCode = readFile("shaders/intrinsic_supporting_vert.spv");
    auto geomShaderCode = readFile("shaders/intrinsic_supporting_geom.spv");
    auto fragShaderCode = readFile("shaders/intrinsic_supporting_frag.spv");

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge vertex shader module" << std::endl;
        return false;
    }
    VkShaderModule geomShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, geomShaderCode, geomShaderModule) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge geometry shader module" << std::endl;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        return false;
    }
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge fragment shader module" << std::endl;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo geomShaderStageInfo{};
    geomShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomShaderStageInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    geomShaderStageInfo.module = geomShaderModule;
    geomShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, geomShaderStageInfo, fragShaderStageInfo};

    auto bindingDescriptions = Vertex::getBindingDescriptions();
    auto vertexAttributes = Vertex::getVertexAttributes();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_TRUE;

    VkStencilOpState stencilOp{};
    stencilOp.passOp = VK_STENCIL_OP_REPLACE;
    stencilOp.failOp = VK_STENCIL_OP_KEEP;
    stencilOp.depthFailOp = VK_STENCIL_OP_KEEP;
    stencilOp.compareOp = VK_COMPARE_OP_ALWAYS;
    stencilOp.compareMask = 0xFF;
    stencilOp.writeMask = 0xFF;
    depthStencil.front = stencilOp;
    depthStencil.back = stencilOp;

    VkPipelineColorBlendAttachmentState colorBlendAttachments[4] = {};
    for (int i = 0; i < 4; ++i) {
        colorBlendAttachments[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachments[i].blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 4;
    colorBlending.pAttachments = colorBlendAttachments;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SurfacePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &supportingHalfedgeDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr, &supportingHalfedgePipelineLayout) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge pipeline layout" << std::endl;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 3;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = supportingHalfedgePipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpassIndex;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &supportingHalfedgePipeline) != VK_SUCCESS) {
        std::cerr << "[IntrinsicRenderer] Failed to create supporting halfedge pipeline" << std::endl;
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), supportingHalfedgePipelineLayout, nullptr);
        supportingHalfedgePipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return true;
}

bool IntrinsicRenderer::initializePick(VkRenderPass renderPass, uint32_t subpassIndex) {
    if (pickInitialized) {
        return true;
    }
    if (!surfaceInitialized || supportingHalfedgePipelineLayout == VK_NULL_HANDLE) {
        return false;
    }
    if (!createElementPickPipelines(renderPass, subpassIndex)) {
        cleanupPick();
        return false;
    }
    pickInitialized = true;
    return true;
}

bool IntrinsicRenderer::createElementPickPipelines(
    VkRenderPass renderPass,
    uint32_t subpassIndex) {
    auto vertShaderCode = readFile("shaders/intrinsic_supporting_vert.spv");
    auto geomShaderCode = readFile("shaders/intrinsic_supporting_geom.spv");

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS) {
        return false;
    }
    VkShaderModule geomShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, geomShaderCode, geomShaderModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        return false;
    }

    const bool created =
        createElementPickPipeline(
            renderPass, subpassIndex, vertShaderModule, geomShaderModule,
            "shaders/intrinsic_face_pick_frag.spv", intrinsicFacePickPipeline) &&
        createElementPickPipeline(
            renderPass, subpassIndex, vertShaderModule, geomShaderModule,
            "shaders/intrinsic_vertex_pick_frag.spv", intrinsicVertexPickPipeline) &&
        createElementPickPipeline(
            renderPass, subpassIndex, vertShaderModule, geomShaderModule,
            "shaders/intrinsic_edge_pick_frag.spv", intrinsicEdgePickPipeline);

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
    return created;
}

bool IntrinsicRenderer::createElementPickPipeline(
    VkRenderPass renderPass,
    uint32_t subpassIndex,
    VkShaderModule vertShaderModule,
    VkShaderModule geomShaderModule,
    const char* fragmentShaderPath,
    VkPipeline& outPipeline) {
    auto fragShaderCode = readFile(fragmentShaderPath);
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[1].module = geomShaderModule;
    stages[1].pName = "main";
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[2].module = fragShaderModule;
    stages[2].pName = "main";

    auto bindingDescriptions = Vertex::getBindingDescriptions();
    auto vertexAttributes = Vertex::getVertexAttributes();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInput.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexInput.pVertexAttributeDescriptions = vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 3;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterizer;
    info.pMultisampleState = &multisampling;
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &colorBlending;
    info.pDynamicState = &dynamicState;
    info.layout = supportingHalfedgePipelineLayout;
    info.renderPass = renderPass;
    info.subpass = subpassIndex;

    const VkResult result = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &info, nullptr,
        &outPipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return result == VK_SUCCESS;
}

void IntrinsicRenderer::cleanupPick() {
    if (vulkanDevice.getDevice() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
    }
    if (intrinsicFacePickPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), intrinsicFacePickPipeline, nullptr);
        intrinsicFacePickPipeline = VK_NULL_HANDLE;
    }
    if (intrinsicVertexPickPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), intrinsicVertexPickPipeline, nullptr);
        intrinsicVertexPickPipeline = VK_NULL_HANDLE;
    }
    if (intrinsicEdgePickPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), intrinsicEdgePickPipeline, nullptr);
        intrinsicEdgePickPipeline = VK_NULL_HANDLE;
    }
    pickInitialized = false;
}

bool IntrinsicRenderer::surfaceMatches(
    uint64_t socketKey,
    uint64_t surfaceIdentity) const {
    const auto it = remeshConfigsBySocket.find(socketKey);
    return it != remeshConfigsBySocket.end() &&
        it->second.isValid() &&
        it->second.surfaceIdentity == surfaceIdentity;
}

void IntrinsicRenderer::renderElementPick(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    uint64_t socketKey,
    uint64_t surfaceIdentity,
    MeshElementType elementType) {
    if (!pickInitialized || currentFrame >= maxFramesInFlight) {
        return;
    }
    const auto configIt = remeshConfigsBySocket.find(socketKey);
    if (configIt == remeshConfigsBySocket.end()) {
        return;
    }
    const RemeshDisplayController::Config& config = configIt->second;
    if (!config.isValid() || config.surfaceIdentity != surfaceIdentity) {
        return;
    }

    VkPipeline pipeline = VK_NULL_HANDLE;
    float selectionType = 0.0f;
    switch (elementType) {
    case MeshElementType::Vertex:
        if (config.intrinsicVertexBuffer == VK_NULL_HANDLE) {
            return;
        }
        pipeline = intrinsicVertexPickPipeline;
        selectionType = 0.0f;
        break;
    case MeshElementType::Edge:
        pipeline = intrinsicEdgePickPipeline;
        selectionType = 1.0f;
        break;
    case MeshElementType::Face:
    default:
        if (config.faceIdView == VK_NULL_HANDLE) {
            return;
        }
        pipeline = intrinsicFacePickPipeline;
        selectionType = 2.0f;
        break;
    }
    if (pipeline == VK_NULL_HANDLE) {
        return;
    }

    const auto descriptorIt =
        supportingHalfedgeDescriptorSetsBySocket.find(socketKey);
    if (descriptorIt == supportingHalfedgeDescriptorSetsBySocket.end() ||
        currentFrame >= descriptorIt->second.size()) {
        return;
    }

    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        supportingHalfedgePipelineLayout, 0, 1,
        &descriptorIt->second[currentFrame], 0, nullptr);

    SurfacePushConstants pushConstants{
        resolveModelMatrix(config),
        glm::vec4(0.0f, 0.0f, viewportSize.x, viewportSize.y),
        glm::vec4(
            selectionType,
            resolveVertexMarkerRadius(config),
            12.0f,
            0.0f)};
    vkCmdPushConstants(
        commandBuffer, supportingHalfedgePipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(pushConstants), &pushConstants);

    VkBuffer vertexBuffer = config.renderVertexBuffer;
    VkDeviceSize vertexOffset = config.renderVertexBufferOffset;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(
        commandBuffer, config.renderIndexBuffer,
        config.renderIndexBufferOffset, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, config.renderIndexCount, 1, 0, 0, 0);
}

bool IntrinsicRenderer::createNormalPipeline(
    VkRenderPass renderPass,
    uint32_t subpassIndex,
    const char* vertexShaderPath,
    const char* geometryShaderPath,
    const char* fragmentShaderPath,
    VkPipeline& outPipeline) {
    auto vertShaderCode = readFile(vertexShaderPath);
    auto geomShaderCode = readFile(geometryShaderPath);
    auto fragShaderCode = readFile(fragmentShaderPath);

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS) {
        return false;
    }
    VkShaderModule geomShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, geomShaderCode, geomShaderModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        return false;
    }
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShaderModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo geomStage{};
    geomStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    geomStage.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    geomStage.module = geomShaderModule;
    geomStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShaderModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, geomStage, fragStage};

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 2.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 3;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = normalPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpassIndex;

    const VkResult result = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
        &outPipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return result == VK_SUCCESS;
}

bool IntrinsicRenderer::createXrayWireframePipeline(
    VkRenderPass renderPass,
    uint32_t subpassIndex) {
    auto vertShaderCode = readFile("shaders/intrinsic_supporting_vert.spv");
    auto geomShaderCode = readFile("shaders/intrinsic_supporting_geom.spv");
    auto fragShaderCode = readFile("shaders/intrinsic_xray_frag.spv");

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS) {
        return false;
    }
    VkShaderModule geomShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, geomShaderCode, geomShaderModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        return false;
    }
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[3]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertShaderModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    stages[1].module = geomShaderModule;
    stages[1].pName = "main";
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[2].module = fragShaderModule;
    stages[2].pName = "main";

    auto bindingDescriptions = Vertex::getBindingDescriptions();
    auto vertexAttributes = Vertex::getVertexAttributes();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindingDescriptions.size());
    vertexInput.pVertexBindingDescriptions = bindingDescriptions.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexInput.pVertexAttributeDescriptions = vertexAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 3;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterizer;
    info.pMultisampleState = &multisampling;
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &colorBlending;
    info.pDynamicState = &dynamicState;
    info.layout = supportingHalfedgePipelineLayout;
    info.renderPass = renderPass;
    info.subpass = subpassIndex;

    const VkResult result = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &info, nullptr,
        &xrayWireframePipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return result == VK_SUCCESS;
}

void IntrinsicRenderer::renderGeometry(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    int wireframeMode,
    const ModelSelection* modelSelection) {
    if (!surfaceInitialized) {
        return;
    }
    drawIntrinsicSurfaces(
        commandBuffer,
        currentFrame,
        supportingHalfedgePipeline,
        wireframeMode,
        true,
        true,
        modelSelection);
}

void IntrinsicRenderer::renderXRayWireframe(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    const ModelSelection& modelSelection) {
    if (!surfaceInitialized || !overlayInitialized) {
        return;
    }
    drawIntrinsicSurfaces(
        commandBuffer,
        currentFrame,
        xrayWireframePipeline,
        1,
        false,
        false,
        &modelSelection);
}

void IntrinsicRenderer::drawIntrinsicSurfaces(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    VkPipeline pipeline,
    int mode,
    bool applySelection,
    bool setStencil,
    const ModelSelection* modelSelection) {
    if (pipeline == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (const auto& [surfaceIdentity, config] : remeshConfigsBySocket) {
        if (!config.isValid()) {
            continue;
        }

        bool selectionActive = false;
        MeshElementType selectionType = MeshElementType::Face;
        if (applySelection) {
            const auto selectionIt =
                selectionConfigsBySocket.find(surfaceIdentity);
            selectionActive =
                selectionIt != selectionConfigsBySocket.end() &&
                selectionIt->second.surfaceIdentity == config.surfaceIdentity &&
                selectionIt->second.selectedCount != 0 &&
                !selectionIt->second.elementMask.empty() &&
                elementMaskView(surfaceIdentity, currentFrame) != zeroElementMaskView;
            if (selectionActive) {
                selectionType = selectionIt->second.elementType;
            }
        }

        auto it = supportingHalfedgeDescriptorSetsBySocket.find(surfaceIdentity);
        if (it == supportingHalfedgeDescriptorSetsBySocket.end()) {
            continue;
        }
        const auto& modelDescriptorSets = it->second;
        if (currentFrame >= modelDescriptorSets.size()) {
            continue;
        }

        const glm::mat4 modelMatrix = resolveModelMatrix(config);
        const glm::vec3 wireframeColor =
            modelSelection &&
            modelSelection->isModelSelected(config.runtimeModelId)
                ? renderconfig::SelectedWireframeColor
                : renderconfig::WireframeColor;
        const float vertexHighlightRadius =
            resolveVertexMarkerRadius(config);
        SurfacePushConstants pushConstants{
            modelMatrix,
            glm::vec4(
                0.0f,
                selectionActive ? 1.0f : 0.0f,
                viewportSize.x,
                viewportSize.y),
            glm::vec4(
                static_cast<float>(selectionType),
                vertexHighlightRadius,
                EdgeHighlightHalfWidthPixels,
                static_cast<float>(mode)),
            glm::vec4(wireframeColor, 1.0f)
        };
        vkCmdPushConstants(
            commandBuffer,
            supportingHalfedgePipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(pushConstants),
            &pushConstants);

        if (setStencil) {
            vkCmdSetStencilReference(
                commandBuffer,
                VK_STENCIL_FACE_FRONT_AND_BACK,
                config.runtimeModelId);
        }

        VkBuffer modelVertexBuffer = config.renderVertexBuffer;
        VkDeviceSize modelVertexOffset = config.renderVertexBufferOffset;
        vkCmdBindVertexBuffers(
            commandBuffer, 0, 1, &modelVertexBuffer, &modelVertexOffset);
        vkCmdBindIndexBuffer(
            commandBuffer, config.renderIndexBuffer,
            config.renderIndexBufferOffset, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            supportingHalfedgePipelineLayout, 0, 1,
            &modelDescriptorSets[currentFrame], 0, nullptr);
        vkCmdDrawIndexed(
            commandBuffer, config.renderIndexCount, 1, 0, 0, 0);
    }
}

void IntrinsicRenderer::renderOverlay(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
    renderNormals(
        commandBuffer, currentFrame,
        intrinsicNormalsPipeline,
        intrinsicNormalsDescriptorSetsBySocket,
        true);
    renderNormals(
        commandBuffer, currentFrame,
        intrinsicVertexNormalsPipeline,
        intrinsicVertexNormalsDescriptorSetsBySocket,
        false);
}

void IntrinsicRenderer::renderNormals(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    VkPipeline pipeline,
    const std::unordered_map<uint64_t, std::vector<VkDescriptorSet>>& descriptorSetsBySocket,
    bool faceNormals) {
    if (!overlayInitialized || pipeline == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (const auto& [socketKey, config] : remeshConfigsBySocket) {
        if (!config.isValid() ||
            (faceNormals ? !config.showFaceNormals : !config.showVertexNormals)) {
            continue;
        }
        const auto it = descriptorSetsBySocket.find(socketKey);
        if (it == descriptorSetsBySocket.end() ||
            currentFrame >= it->second.size()) {
            continue;
        }

        const size_t count = faceNormals
            ? config.intrinsicTriangleCount
            : config.intrinsicVertexCount;
        if (count == 0) {
            continue;
        }

        NormalPushConstant pushConstants{};
        pushConstants.modelMatrix = resolveModelMatrix(config);
        pushConstants.normalLength = config.normalLength;
        pushConstants.avgArea = faceNormals
            ? config.averageTriangleArea
            : 0.0f;

        vkCmdPushConstants(
            commandBuffer, normalPipelineLayout,
            VK_SHADER_STAGE_GEOMETRY_BIT,
            0, sizeof(NormalPushConstant), &pushConstants);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            normalPipelineLayout, 0, 1,
            &it->second[currentFrame], 0, nullptr);
        vkCmdDraw(
            commandBuffer, static_cast<uint32_t>(count), 1, 0, 0);
    }
}

void IntrinsicRenderer::cleanup() {
    cleanupOverlay();
    cleanupPick();

    if (supportingHalfedgePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), supportingHalfedgePipeline, nullptr);
        supportingHalfedgePipeline = VK_NULL_HANDLE;
    }

    for (auto& [surfaceIdentity, buffers] : elementMaskBuffersBySocket) {
        (void)surfaceIdentity;
        for (ElementMaskBuffer& buffer : buffers) {
            if (buffer.view != VK_NULL_HANDLE) {
                vkDestroyBufferView(vulkanDevice.getDevice(), buffer.view, nullptr);
            }
            if (buffer.buffer != VK_NULL_HANDLE) {
                allocator.free(buffer.buffer, buffer.offset);
            }
        }
    }
    elementMaskBuffersBySocket.clear();
    if (zeroElementMaskView != VK_NULL_HANDLE) {
        vkDestroyBufferView(vulkanDevice.getDevice(), zeroElementMaskView, nullptr);
        zeroElementMaskView = VK_NULL_HANDLE;
    }
    if (zeroElementMaskBuffer != VK_NULL_HANDLE) {
        allocator.free(zeroElementMaskBuffer, zeroElementMaskOffset);
        zeroElementMaskBuffer = VK_NULL_HANDLE;
    }

    if (supportingHalfedgePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), supportingHalfedgePipelineLayout, nullptr);
        supportingHalfedgePipelineLayout = VK_NULL_HANDLE;
    }

    if (supportingHalfedgeDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), supportingHalfedgeDescriptorSetLayout, nullptr);
        supportingHalfedgeDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (supportingHalfedgeDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), supportingHalfedgeDescriptorPool, nullptr);
        supportingHalfedgeDescriptorPool = VK_NULL_HANDLE;
    }

    if (wireframeTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(vulkanDevice.getDevice(), wireframeTextureSampler, nullptr);
        wireframeTextureSampler = VK_NULL_HANDLE;
    }
    if (wireframeTextureView != VK_NULL_HANDLE) {
        vkDestroyImageView(vulkanDevice.getDevice(), wireframeTextureView, nullptr);
        wireframeTextureView = VK_NULL_HANDLE;
    }
    if (wireframeTextureImage != VK_NULL_HANDLE) {
        vkDestroyImage(vulkanDevice.getDevice(), wireframeTextureImage, nullptr);
        wireframeTextureImage = VK_NULL_HANDLE;
    }
    if (wireframeTextureMemory != VK_NULL_HANDLE) {
        allocator.freeImageMemory(wireframeTextureMemory);
        wireframeTextureMemory = VK_NULL_HANDLE;
    }

    supportingHalfedgeDescriptorSetsBySocket.clear();
    remeshConfigsBySocket.clear();
    selectionConfigsBySocket.clear();
    surfaceInitialized = false;
}

