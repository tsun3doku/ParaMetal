#include "HeatSystemSurfaceStage.hpp"

#include "HeatReceiverRuntime.hpp"
#include "HeatSystemResources.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "scene/Model.hpp"

#include "util/Structs.hpp"
#include "util/file_utils.h"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <array>
#include <iostream>
#include <vector>

HeatSystemSurfaceStage::HeatSystemSurfaceStage(const HeatSystemStageContext& stageContext)
    : context(stageContext) {
}

void HeatSystemSurfaceStage::refreshSurfaceDescriptors(uint32_t nodeCount) {
    (void)nodeCount;
}

bool HeatSystemSurfaceStage::createDescriptorPool(uint32_t maxFramesInFlight) {
    (void)maxFramesInFlight;
    const uint32_t maxHeatModels = 10;
    const uint32_t totalSets = (maxHeatModels * 2);

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = totalSets * 6;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = totalSets * 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = totalSets;

    if (vkCreateDescriptorPool(context.vulkanDevice.getDevice(), &poolInfo, nullptr,
        &context.resources.surfaceDescriptorPool) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create surface descriptor pool" << std::endl;
        return false;
    }
    return true;
}

bool HeatSystemSurfaceStage::createDescriptorSetLayout() {
    // Layout for temperature pass (bindings 0, 1, 10, 11)
    std::vector<VkDescriptorSetLayoutBinding> tempBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo tempBindingFlags{};
    tempBindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

    std::vector<VkDescriptorBindingFlags> tempFlags(tempBindings.size(),
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);

    tempBindingFlags.bindingCount = static_cast<uint32_t>(tempFlags.size());
    tempBindingFlags.pBindingFlags = tempFlags.data();

    VkDescriptorSetLayoutCreateInfo tempLayoutInfo{};
    tempLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tempLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    tempLayoutInfo.bindingCount = static_cast<uint32_t>(tempBindings.size());
    tempLayoutInfo.pBindings = tempBindings.data();
    tempLayoutInfo.pNext = &tempBindingFlags;

    if (vkCreateDescriptorSetLayout(context.vulkanDevice.getDevice(), &tempLayoutInfo, nullptr,
        &context.resources.surfaceDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create surface temperature descriptor set layout" << std::endl;
        return false;
    }

    // Layout for gradient pass (bindings 0, 1, 2, 10, 12)
    std::vector<VkDescriptorSetLayoutBinding> gradientBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo gradientBindingFlags{};
    gradientBindingFlags.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

    std::vector<VkDescriptorBindingFlags> gradientFlags(gradientBindings.size(),
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);

    gradientBindingFlags.bindingCount = static_cast<uint32_t>(gradientFlags.size());
    gradientBindingFlags.pBindingFlags = gradientFlags.data();

    VkDescriptorSetLayoutCreateInfo gradientLayoutInfo{};
    gradientLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    gradientLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    gradientLayoutInfo.bindingCount = static_cast<uint32_t>(gradientBindings.size());
    gradientLayoutInfo.pBindings = gradientBindings.data();
    gradientLayoutInfo.pNext = &gradientBindingFlags;

    if (vkCreateDescriptorSetLayout(context.vulkanDevice.getDevice(), &gradientLayoutInfo, nullptr,
        &context.resources.surfaceGradientDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create surface gradient descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool HeatSystemSurfaceStage::createPipeline() {
    // Temperature pipeline 
    auto tempShaderCode = readFile("shaders/heat_surface_temp_comp.spv");
    VkShaderModule tempShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(context.vulkanDevice, tempShaderCode, tempShaderModule) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create surface temperature compute shader module" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo tempShaderStageInfo{};
    tempShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tempShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tempShaderStageInfo.module = tempShaderModule;
    tempShaderStageInfo.pName = "main";

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(heat::SourcePushConstant);

    VkPipelineLayoutCreateInfo tempPipelineLayoutInfo{};
    tempPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tempPipelineLayoutInfo.setLayoutCount = 1;
    tempPipelineLayoutInfo.pSetLayouts = &context.resources.surfaceDescriptorSetLayout;
    tempPipelineLayoutInfo.pushConstantRangeCount = 1;
    tempPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(context.vulkanDevice.getDevice(), &tempPipelineLayoutInfo, nullptr,
        &context.resources.surfacePipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), tempShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create surface temperature pipeline layout" << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo tempPipelineInfo{};
    tempPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    tempPipelineInfo.stage = tempShaderStageInfo;
    tempPipelineInfo.layout = context.resources.surfacePipelineLayout;

    if (vkCreateComputePipelines(context.vulkanDevice.getDevice(), VK_NULL_HANDLE, 1,
        &tempPipelineInfo, nullptr, &context.resources.surfacePipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.vulkanDevice.getDevice(), context.resources.surfacePipelineLayout, nullptr);
        context.resources.surfacePipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), tempShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create surface temperature compute pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(context.vulkanDevice.getDevice(), tempShaderModule, nullptr);

    // Gradient pipeline 
    auto gradientShaderCode = readFile("shaders/heat_surface_gradient_comp.spv");
    VkShaderModule gradientShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(context.vulkanDevice, gradientShaderCode, gradientShaderModule) != VK_SUCCESS) {
        std::cerr << "[HeatSystem] Failed to create surface gradient compute shader module" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo gradientShaderStageInfo{};
    gradientShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    gradientShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    gradientShaderStageInfo.module = gradientShaderModule;
    gradientShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo gradientPipelineLayoutInfo{};
    gradientPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gradientPipelineLayoutInfo.setLayoutCount = 1;
    gradientPipelineLayoutInfo.pSetLayouts = &context.resources.surfaceGradientDescriptorSetLayout;
    gradientPipelineLayoutInfo.pushConstantRangeCount = 1;
    gradientPipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(context.vulkanDevice.getDevice(), &gradientPipelineLayoutInfo, nullptr,
        &context.resources.surfaceGradientPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), gradientShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create surface gradient pipeline layout" << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo gradientPipelineInfo{};
    gradientPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    gradientPipelineInfo.stage = gradientShaderStageInfo;
    gradientPipelineInfo.layout = context.resources.surfaceGradientPipelineLayout;

    if (vkCreateComputePipelines(context.vulkanDevice.getDevice(), VK_NULL_HANDLE, 1,
        &gradientPipelineInfo, nullptr, &context.resources.surfaceGradientPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(context.vulkanDevice.getDevice(), context.resources.surfaceGradientPipelineLayout, nullptr);
        context.resources.surfaceGradientPipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(context.vulkanDevice.getDevice(), gradientShaderModule, nullptr);
        std::cerr << "[HeatSystem] Failed to create surface gradient compute pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(context.vulkanDevice.getDevice(), gradientShaderModule, nullptr);
    return true;
}

void HeatSystemSurfaceStage::dispatchSurfaceTemperatureUpdates(
    VkCommandBuffer commandBuffer,
    uint32_t nodeCount,
    const std::vector<std::unique_ptr<HeatReceiverRuntime>>& receivers,
    const std::vector<VkDescriptorSet>& surfaceComputeSetsA,
    const std::vector<VkDescriptorSet>& surfaceComputeSetsB,
    bool finalWritesBufferB) const {
    if (nodeCount == 0 ||
        context.resources.surfacePipeline == VK_NULL_HANDLE ||
        context.resources.surfacePipelineLayout == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.resources.surfacePipeline);

    heat::SourcePushConstant surfacePushConstant{};
    surfacePushConstant.substepIndex = 0;

    for (size_t i = 0; i < receivers.size(); ++i) {
        const auto& receiver = receivers[i];
        if (!receiver || receiver->getIntrinsicVertexCount() == 0) {
            continue;
        }

        const VkDescriptorSet surfaceSet = finalWritesBufferB
            ? (i < surfaceComputeSetsB.size() ? surfaceComputeSetsB[i] : VK_NULL_HANDLE)
            : (i < surfaceComputeSetsA.size() ? surfaceComputeSetsA[i] : VK_NULL_HANDLE);
        if (surfaceSet == VK_NULL_HANDLE) {
            continue;
        }

        const uint32_t workGroupSize = 256;
        const uint32_t vertexCount = static_cast<uint32_t>(receiver->getIntrinsicVertexCount());
        const uint32_t workGroupCount = (vertexCount + workGroupSize - 1) / workGroupSize;

        vkCmdPushConstants(
            commandBuffer,
            context.resources.surfacePipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(heat::SourcePushConstant),
            &surfacePushConstant);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            context.resources.surfacePipelineLayout,
            0,
            1,
            &surfaceSet,
            0,
            nullptr);
        vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
    }
}

void HeatSystemSurfaceStage::dispatchSurfaceGradientUpdates(
    VkCommandBuffer commandBuffer,
    uint32_t nodeCount,
    const std::vector<std::unique_ptr<HeatReceiverRuntime>>& receivers,
    const std::vector<VkDescriptorSet>& surfaceGradientComputeSetsA,
    const std::vector<VkDescriptorSet>& surfaceGradientComputeSetsB,
    bool finalWritesBufferB) const {
    if (nodeCount == 0 ||
        context.resources.surfaceGradientPipeline == VK_NULL_HANDLE ||
        context.resources.surfaceGradientPipelineLayout == VK_NULL_HANDLE) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, context.resources.surfaceGradientPipeline);

    heat::SourcePushConstant surfacePushConstant{};
    surfacePushConstant.substepIndex = 0;

    for (size_t i = 0; i < receivers.size(); ++i) {
        const auto& receiver = receivers[i];
        if (!receiver || receiver->getIntrinsicVertexCount() == 0) {
            continue;
        }

        const VkDescriptorSet gradientSet = finalWritesBufferB
            ? (i < surfaceGradientComputeSetsB.size() ? surfaceGradientComputeSetsB[i] : VK_NULL_HANDLE)
            : (i < surfaceGradientComputeSetsA.size() ? surfaceGradientComputeSetsA[i] : VK_NULL_HANDLE);
        if (gradientSet == VK_NULL_HANDLE) {
            continue;
        }

        const uint32_t workGroupSize = 256;
        const uint32_t vertexCount = static_cast<uint32_t>(receiver->getIntrinsicVertexCount());
        const uint32_t workGroupCount = (vertexCount + workGroupSize - 1) / workGroupSize;

        vkCmdPushConstants(
            commandBuffer,
            context.resources.surfaceGradientPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(heat::SourcePushConstant),
            &surfacePushConstant);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            context.resources.surfaceGradientPipelineLayout,
            0,
            1,
            &gradientSet,
            0,
            nullptr);
        vkCmdDispatch(commandBuffer, workGroupCount, 1, 1);
    }
}
