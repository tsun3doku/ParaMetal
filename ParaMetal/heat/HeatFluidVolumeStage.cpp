#include "HeatFluidVolumeStage.hpp"

#include "util/file_utils.h"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/MemoryAllocator.hpp"

#include <array>
#include <iostream>

HeatFluidVolumeStage::HeatFluidVolumeStage(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

HeatFluidVolumeStage::~HeatFluidVolumeStage() {
    cleanup();
}

void HeatFluidVolumeStage::cleanup() {
    ready = false;
    VkDevice device = vulkanDevice.getDevice();

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    descriptorSets.clear();

    if (seedToFluidNodeBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedToFluidNodeBuffer, seedToFluidNodeBufferOffset);
        seedToFluidNodeBuffer = VK_NULL_HANDLE;
        seedToFluidNodeBufferOffset = 0;
        seedToFluidNodeCount = 0;
    }

    if (temperatureVolumeSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, temperatureVolumeSampler, nullptr);
        temperatureVolumeSampler = VK_NULL_HANDLE;
    }
    if (temperatureVolumeView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, temperatureVolumeView, nullptr);
        temperatureVolumeView = VK_NULL_HANDLE;
    }
    if (temperatureVolumeImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, temperatureVolumeImage, nullptr);
        temperatureVolumeImage = VK_NULL_HANDLE;
    }
    if (temperatureVolumeMemory != VK_NULL_HANDLE) {
        memoryAllocator.freeImageMemory(temperatureVolumeMemory);
        temperatureVolumeMemory = VK_NULL_HANDLE;
    }
    gridDimensions = glm::ivec3(0);
    temperatureVolumeLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

bool HeatFluidVolumeStage::initialize(const glm::ivec3& gridDim) {
    if (gridDim.x < 2 || gridDim.y < 2 || gridDim.z < 2) {
        return false;
    }

    if (ready && gridDimensions == gridDim) {
        return true;
    }

    cleanup();

    if (!createPipeline()) {
        cleanup();
        return false;
    }

    if (!createVolumeResources(gridDim)) {
        cleanup();
        return false;
    }

    gridDimensions = gridDim;
    ready = true;
    return true;
}

bool HeatFluidVolumeStage::createPipeline() {
    VkDevice device = vulkanDevice.getDevice();

    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create descriptor set layout." << std::endl;
        return false;
    }

    const uint32_t setCount = (maxFramesInFlight > 0) ? maxFramesInFlight : 1;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * setCount};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 * setCount};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create descriptor pool." << std::endl;
        return false;
    }

    std::vector<VkDescriptorSetLayout> layouts(setCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(setCount);
    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to allocate descriptor sets." << std::endl;
        return false;
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create pipeline layout." << std::endl;
        return false;
    }

    auto shaderCode = readFile("shaders/heat_fluid_volume_comp.spv");
    if (shaderCode.empty()) {
        std::cerr << "[HeatFluidVolumeStage] Failed to read shaders/heat_fluid_volume_comp.spv" << std::endl;
        return false;
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, shaderCode, shaderModule) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create shader module." << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineInfo.stage.module = shaderModule;
    computePipelineInfo.stage.pName = "main";
    computePipelineInfo.layout = pipelineLayout;

    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (res != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create compute pipeline." << std::endl;
        return false;
    }

    return true;
}

bool HeatFluidVolumeStage::createVolumeResources(const glm::ivec3& gridDim) {
    VkResult res = createImage3D(
        vulkanDevice,
        memoryAllocator,
        static_cast<uint32_t>(gridDim.x),
        static_cast<uint32_t>(gridDim.y),
        static_cast<uint32_t>(gridDim.z),
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        temperatureVolumeImage,
        temperatureVolumeMemory);

    if (res != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create 3D image." << std::endl;
        return false;
    }

    res = createImageView3D(
        vulkanDevice,
        temperatureVolumeImage,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        temperatureVolumeView);

    if (res != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create 3D image view." << std::endl;
        return false;
    }

    temperatureVolumeLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &temperatureVolumeSampler) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to create NEAREST sampler." << std::endl;
        return false;
    }

    return true;
}

bool HeatFluidVolumeStage::updateSeedToFluidNode(
    const std::vector<glm::vec3>& seeds,
    const topology::GlobalDomain& globalDomain,
    const std::vector<uint32_t>& globalFragmentToFluidNode)
{
    if (seeds.empty() || !globalDomain.isValid()) {
        return false;
    }

    std::vector<uint32_t> seedToFluid(seeds.size(), 0xFFFFFFFFu);
    if (!globalDomain.seedToFragment.empty()) {
        for (size_t s = 0; s < seeds.size() && s < globalDomain.seedToFragment.size(); ++s) {
            const uint32_t fragment = globalDomain.seedToFragment[s];
            if (fragment < globalDomain.fragments.size() &&
                globalDomain.fragments[fragment].instanceId == topology::InvalidInstanceId &&
                fragment < globalFragmentToFluidNode.size()) {
                seedToFluid[s] = globalFragmentToFluidNode[fragment];
            }
        }
    } else {
        for (size_t f = 0; f < globalDomain.fragments.size(); ++f) {
            const auto& frag = globalDomain.fragments[f];
            if (frag.instanceId == topology::InvalidInstanceId && f < globalFragmentToFluidNode.size()) {
                const uint32_t fluidNode = globalFragmentToFluidNode[f];
                const uint32_t seedId = frag.seedId;
                if (seedId < seedToFluid.size()) {
                    seedToFluid[seedId] = fluidNode;
                }
            }
        }
    }

    if (seedToFluidNodeBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedToFluidNodeBuffer, seedToFluidNodeBufferOffset);
        seedToFluidNodeBuffer = VK_NULL_HANDLE;
        seedToFluidNodeBufferOffset = 0;
        seedToFluidNodeCount = 0;
    }

    VkBuffer newBuffer = VK_NULL_HANDLE;
    VkDeviceSize newOffset = 0;
    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            seedToFluid.data(),
            seedToFluid.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sizeof(uint32_t),
            newBuffer,
            newOffset) != VK_SUCCESS) {
        std::cerr << "[HeatFluidVolumeStage] Failed to upload seedToFluidNode buffer." << std::endl;
        return false;
    }

    seedToFluidNodeBuffer = newBuffer;
    seedToFluidNodeBufferOffset = newOffset;
    seedToFluidNodeCount = static_cast<uint32_t>(seedToFluid.size());
    return true;
}

void HeatFluidVolumeStage::recordCompute(
    VkCommandBuffer cmd,
    uint32_t currentFrame,
    VkBuffer fluidSeedGridBuffer,
    VkDeviceSize fluidSeedGridOffset,
    VkBuffer temperatureSourceBuffer,
    VkDeviceSize temperatureSourceOffset)
{
    if (!ready ||
        descriptorSets.empty() ||
        fluidSeedGridBuffer == VK_NULL_HANDLE ||
        seedToFluidNodeBuffer == VK_NULL_HANDLE ||
        temperatureSourceBuffer == VK_NULL_HANDLE ||
        temperatureVolumeView == VK_NULL_HANDLE) {
        return;
    }

    const uint32_t frameIdx = currentFrame % static_cast<uint32_t>(descriptorSets.size());
    VkDescriptorSet currentSet = descriptorSets[frameIdx];

    VkDescriptorBufferInfo gridInfo{};
    gridInfo.buffer = fluidSeedGridBuffer;
    gridInfo.offset = fluidSeedGridOffset;
    gridInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo seedMapInfo{};
    seedMapInfo.buffer = seedToFluidNodeBuffer;
    seedMapInfo.offset = seedToFluidNodeBufferOffset;
    seedMapInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo tempInfo{};
    tempInfo.buffer = temperatureSourceBuffer;
    tempInfo.offset = temperatureSourceOffset;
    tempInfo.range = VK_WHOLE_SIZE;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = temperatureVolumeView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &gridInfo, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &seedMapInfo, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &tempInfo, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr};

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &currentSet, 0, nullptr);

    // Invariant: The volume compute and graphics consumption rely on the same ordered submission queue.
    VkImageMemoryBarrier beforeWrite{};
    beforeWrite.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    beforeWrite.oldLayout = temperatureVolumeLayout;
    beforeWrite.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    beforeWrite.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeWrite.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    beforeWrite.image = temperatureVolumeImage;
    beforeWrite.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkPipelineStageFlags srcStage;
    if (temperatureVolumeLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
        beforeWrite.srcAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    } else {
        beforeWrite.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    beforeWrite.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmd,
        srcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &beforeWrite);

    const uint32_t groupX = (static_cast<uint32_t>(gridDimensions.x) + 7) / 8;
    const uint32_t groupY = (static_cast<uint32_t>(gridDimensions.y) + 7) / 8;
    const uint32_t groupZ = (static_cast<uint32_t>(gridDimensions.z) + 7) / 8;
    vkCmdDispatch(cmd, groupX, groupY, groupZ);

    VkImageMemoryBarrier afterWrite{};
    afterWrite.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    afterWrite.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    afterWrite.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    afterWrite.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    afterWrite.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    afterWrite.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    afterWrite.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    afterWrite.image = temperatureVolumeImage;
    afterWrite.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &afterWrite);

    temperatureVolumeLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
