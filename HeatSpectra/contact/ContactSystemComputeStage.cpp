#include "ContactSystemComputeStage.hpp"

#include "contact/ContactGpuStructs.hpp"

#include "util/file_utils.h"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <array>
#include <iostream>

namespace {
struct ContactPushConstant {
    uint32_t elementCount = 0;
};
}

ContactSystemComputeStage::ContactSystemComputeStage(VulkanDevice& device)
    : vulkanDevice(device) {
}

ContactSystemComputeStage::~ContactSystemComputeStage() {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
}

void ContactSystemComputeStage::dispatchGather(
    VkCommandBuffer commandBuffer,
    VkDescriptorSet descriptorSet,
    uint32_t elementCount) const {

    if (pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE || descriptorSet == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    ContactPushConstant pushConst{};
    pushConst.elementCount = elementCount;

    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(ContactPushConstant),
        &pushConst);

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);

    vkCmdDispatch(commandBuffer, (elementCount + 255) / 256, 1, 1);
}

bool ContactSystemComputeStage::createDescriptorPool(uint32_t maxSets) {
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    // 4 descriptor sets per contact relationship (A->B/B->A * Ping/Pong)
    uint32_t numSets = maxSets * 4;
    uint32_t storageBufferDescriptors = numSets * 4;

    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = storageBufferDescriptors;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = numSets;

    return vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPool) == VK_SUCCESS;
}

bool ContactSystemComputeStage::createDescriptorSetLayout() {
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // source field buffer
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // dest accumulator buffer
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // contactEdges (ContactSampleWeight)
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // contactIndices (ContactIndex)
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = 0;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    layoutInfo.pNext = nullptr;

    return vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) == VK_SUCCESS;
}

bool ContactSystemComputeStage::createPipeline() {
    if (descriptorSetLayout == VK_NULL_HANDLE) {
        return false;
    }

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    std::vector<char> computeShaderCode;
    if (!readFile("shaders/contact_comp.spv", computeShaderCode) || computeShaderCode.empty()) {
        std::cerr << "[ContactSystem] Failed to read contact compute shader" << std::endl;
        return false;
    }

    VkShaderModule computeShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, computeShaderCode, computeShaderModule) != VK_SUCCESS) {
        std::cerr << "[ContactSystem] Failed to create Contact compute shader module" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(ContactPushConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), computeShaderModule, nullptr);
        std::cerr << "[ContactSystem] Failed to create Contact pipeline layout" << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    const bool ok = vkCreateComputePipelines(
        vulkanDevice.getDevice(),
        VK_NULL_HANDLE,
        1,
        &pipelineInfo,
        nullptr,
        &pipeline) == VK_SUCCESS;

    vkDestroyShaderModule(vulkanDevice.getDevice(), computeShaderModule, nullptr);

    if (!ok) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        std::cerr << "[ContactSystem] Failed to create Contact compute pipeline" << std::endl;
        return false;
    }

    return true;
}

bool ContactSystemComputeStage::createGatherDescriptorSet(
    VkBuffer sourceFieldBuffer,
    VkDeviceSize sourceFieldOffset,
    uint32_t sourceElementCount,
    VkBuffer destAccumulatorBuffer,
    VkDeviceSize destAccumulatorOffset,
    uint32_t destElementCount,
    VkBuffer edgesBuffer,
    VkDeviceSize edgesOffset,
    uint32_t edgeCount,
    VkBuffer indicesBuffer,
    VkDeviceSize indicesOffset,
    uint32_t indexCount,
    VkDescriptorSet& outSet) const {

    outSet = VK_NULL_HANDLE;
    if (descriptorPool == VK_NULL_HANDLE || descriptorSetLayout == VK_NULL_HANDLE) {
        return false;
    }

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &outSet) != VK_SUCCESS) {
        return false;
    }

    std::array<VkDescriptorBufferInfo, 4> infos{};
    infos[0] = {sourceFieldBuffer, sourceFieldOffset, sourceElementCount * sizeof(float)};
    infos[1] = {destAccumulatorBuffer, destAccumulatorOffset, destElementCount * sizeof(float) * 2};
    infos[2] = {edgesBuffer, edgesOffset, edgeCount * sizeof(contact::ContactSampleWeight)};
    infos[3] = {indicesBuffer, indicesOffset, indexCount * sizeof(contact::ContactIndex)};

    std::array<VkWriteDescriptorSet, 4> writes{};
    for (int i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = outSet;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &infos[i];
    }

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}
