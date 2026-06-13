#include "HeatSystemVoronoiStage.hpp"

#include "HeatSystemResources.hpp"
#include "HeatSystemSimRuntime.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

#include "util/file_utils.h"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <array>
#include <iostream>
#include <vector>

HeatSystemVoronoiStage::HeatSystemVoronoiStage(const HeatSystemStageContext& stageContext)
    : context(stageContext) {
}

void HeatSystemVoronoiStage::dispatchDiffusionSubstep(
    VkCommandBuffer commandBuffer,
    VkDescriptorSet descriptorSet,
    const heat::HeatModelPushConstant& pushConstant,
    uint32_t workGroupCount) const {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.resources.voronoiPipeline);
    vkCmdPushConstants(
        commandBuffer,
        context.resources.voronoiPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(heat::HeatModelPushConstant),
        &pushConstant);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        context.resources.voronoiPipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr);
    vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
}

void HeatSystemVoronoiStage::insertFinalTemperatureBarrier(
    VkCommandBuffer commandBuffer,
    uint32_t numSubsteps,
    VkBuffer bufferA,
    VkDeviceSize offsetA,
    VkBuffer bufferB,
    VkDeviceSize offsetB,
    VkDeviceSize bufferSize) const {
    const bool writesBufferB = finalSubstepWritesBufferB(numSubsteps);
    VkBuffer finalTempBuffer = bufferA;
    VkDeviceSize finalTempOffset = offsetA;
    if (writesBufferB) {
        finalTempBuffer = bufferB;
        finalTempOffset = offsetB;
    }

    VkBufferMemoryBarrier finalTempBarrier{};
    finalTempBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    finalTempBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    finalTempBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    finalTempBarrier.buffer = finalTempBuffer;
    finalTempBarrier.offset = finalTempOffset;
    finalTempBarrier.size = bufferSize;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        1,
        &finalTempBarrier,
        0,
        nullptr);
}

bool HeatSystemVoronoiStage::finalSubstepWritesBufferB(uint32_t numSubsteps) const {
    if (numSubsteps == 0) {
        return false;
    }
    return ((numSubsteps - 1) % 2 == 0);
}

bool HeatSystemVoronoiStage::createDescriptorPool(uint32_t numModels) {
    // 2 descriptor sets per model
    uint32_t numSets = numModels * 2;
    uint32_t storageBufferDescriptors = numSets * 7;  // 0,1,2,4,5,6,7
    uint32_t uniformBufferDescriptors = numSets * 1;  // 3

    std::array<VkDescriptorPoolSize, 2> poolSizes{};

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = storageBufferDescriptors;

    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = uniformBufferDescriptors;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = 0;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = numSets;

    if (vkCreateDescriptorPool(
            context.vulkanDevice.getDevice(),
            &poolInfo,
            nullptr,
            &context.resources.voronoiDescriptorPool) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create Voronoi descriptor pool" << std::endl;
        return false;
    }
    return true;
}

bool HeatSystemVoronoiStage::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // simNodeBuffer
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // gmlsInterfaceBuffer
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // materialNodeBuffer
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // playbackUniform
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // tempRead
        {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // tempWrite
        {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // historyBuffer
        {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // couplingAccumulator
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.flags = 0;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    layoutInfo.pNext = nullptr;

    if (vkCreateDescriptorSetLayout(
            context.vulkanDevice.getDevice(),
            &layoutInfo,
            nullptr,
            &context.resources.voronoiDescriptorSetLayout) != VK_SUCCESS) {
        return false;
    }
    return true;
}

bool HeatSystemVoronoiStage::createPipeline() {
    const auto computeShaderCode = readFile("shaders/heat_voronoi_comp.spv");
    VkShaderModule computeShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(context.vulkanDevice, computeShaderCode, computeShaderModule) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create Voronoi compute shader module" << std::endl;
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
    pushConstantRange.size = sizeof(heat::HeatModelPushConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &context.resources.voronoiDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(
            context.vulkanDevice.getDevice(),
            &pipelineLayoutInfo,
            nullptr,
            &context.resources.voronoiPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), computeShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create Voronoi pipeline layout" << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.layout = context.resources.voronoiPipelineLayout;

    if (vkCreateComputePipelines(
            context.vulkanDevice.getDevice(),
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
            nullptr,
            &context.resources.voronoiPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.vulkanDevice.getDevice(), context.resources.voronoiPipelineLayout, nullptr);
        context.resources.voronoiPipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), computeShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create Voronoi compute pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(context.vulkanDevice.getDevice(), computeShaderModule, nullptr);
    return true;
}
