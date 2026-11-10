#include "CFDStreamlineStage.hpp"

#include "CFDGpuStructs.hpp"
#include "util/file_utils.h"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/CommandBufferManager.hpp"

#include <array>
#include <iostream>

CFDStreamlineStage::CFDStreamlineStage(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {
}

CFDStreamlineStage::~CFDStreamlineStage() {
    cleanup();
}

void CFDStreamlineStage::cleanup() {
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

    if (seedBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedBuffer, seedBufferOffset);
        seedBuffer = VK_NULL_HANDLE;
        seedBufferOffset = 0;
    }
    if (seedToFluidNodeBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, seedToFluidNodeBuffer, seedToFluidNodeBufferOffset);
        seedToFluidNodeBuffer = VK_NULL_HANDLE;
        seedToFluidNodeBufferOffset = 0;
    }
    seedToFluidNodeCount = 0;
    if (outputBuffer != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, outputBuffer, outputOffset);
        outputBuffer = VK_NULL_HANDLE;
        outputOffset = 0;
    }
    seedCount = 0;
}

bool CFDStreamlineStage::initialize(
    const topology::GlobalDomain& domain,
    const std::vector<glm::vec4>& seeds,
    const std::vector<uint32_t>& globalFragmentToFluidNode) {
    if (seeds.empty()) {
        std::cerr << "[CFDStreamlineStage] No streamline seeds; streamlines disabled." << std::endl;
        cleanup();
        return false;
    }
    if (!domain.isValid()) {
        std::cerr << "[CFDStreamlineStage] Invalid simulation domain; streamlines disabled." << std::endl;
        cleanup();
        return false;
    }

    cleanup();

    gridMin = domain.volumeGridMin;
    gridDim = domain.volumeGridDim;
    cellSize = domain.volumeGridCellSize;
    domainMin = domain.domainMin;
    domainMax = domain.domainMax;

    if (!createPipeline()) {
        cleanup();
        return false;
    }
    if (!updateSeedToFluidNode(domain, globalFragmentToFluidNode)) {
        cleanup();
        return false;
    }
    if (!createOutputBuffer(static_cast<uint32_t>(seeds.size()))) {
        cleanup();
        return false;
    }

    const VkDeviceSize seedAlignment =
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            seeds.data(),
            seeds.size() * sizeof(glm::vec4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            seedAlignment,
            seedBuffer,
            seedBufferOffset) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to upload seed buffer." << std::endl;
        cleanup();
        return false;
    }

    seedCount = static_cast<uint32_t>(seeds.size());
    ready = true;
    return true;
}

bool CFDStreamlineStage::updateSeedToFluidNode(
    const topology::GlobalDomain& domain,
    const std::vector<uint32_t>& globalFragmentToFluidNode) {
    const size_t cellCount = domain.seedPositions.size();
    if (domain.cellFragmentOffsets.size() != cellCount + 1 ||
        domain.cellFragmentOffsets.back() != domain.cellFragments.size()) {
        std::cerr << "[CFDStreamlineStage] cell-to-fragment CSR is invalid" << std::endl;
        return false;
    }

    std::vector<uint32_t> seedToFluid(cellCount, 0xFFFFFFFFu);
    for (size_t cell = 0; cell < cellCount; ++cell) {
        uint32_t fluidFragment = 0xFFFFFFFFu;
        for (uint32_t entry = domain.cellFragmentOffsets[cell];
             entry < domain.cellFragmentOffsets[cell + 1]; ++entry) {
            const uint32_t fragment = domain.cellFragments[entry];
            if (fragment >= domain.fragments.size() ||
                domain.fragments[fragment].instanceId != topology::InvalidInstanceId) {
                continue;
            }
            if (fluidFragment != 0xFFFFFFFFu) {
                std::cerr << "[CFDStreamlineStage] seed cell " << cell
                          << " has multiple fluid components; point location "
                             "requires fragment boundary geometry"
                          << std::endl;
                return false;
            }
            fluidFragment = fragment;
        }
        if (fluidFragment < globalFragmentToFluidNode.size()) {
            seedToFluid[cell] = globalFragmentToFluidNode[fluidFragment];
        }
    }

    if (uploadDeviceBuffer(
            memoryAllocator,
            commandPool,
            seedToFluid.data(),
            seedToFluid.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment,
            seedToFluidNodeBuffer,
            seedToFluidNodeBufferOffset) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to upload seedToFluidNode buffer." << std::endl;
        return false;
    }

    seedToFluidNodeCount = static_cast<uint32_t>(seedToFluid.size());
    return true;
}

bool CFDStreamlineStage::createOutputBuffer(uint32_t seeds) {
    constexpr VkDeviceSize vertexStride = sizeof(glm::vec4);
    const VkDeviceSize size =
        static_cast<VkDeviceSize>(seeds) * MaxSegmentsPerLine * 2 * vertexStride;

    auto [buffer, offset] = memoryAllocator.allocate(
        size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment);
    if (buffer == VK_NULL_HANDLE) {
        std::cerr << "[CFDStreamlineStage] Failed to allocate line vertex buffer ("
                  << (size / (1024 * 1024)) << " MB)." << std::endl;
        return false;
    }

    // Zeroed vertices are zero-length line segments, so the buffer is safe to
    // draw before the first integration dispatch.
    VkCommandBuffer cmd = commandPool.beginCommands();
    if (cmd == VK_NULL_HANDLE) {
        memoryAllocator.free(buffer, offset);
        return false;
    }
    vkCmdFillBuffer(cmd, buffer, offset, size, 0u);
    if (!commandPool.endCommands(cmd)) {
        memoryAllocator.free(buffer, offset);
        return false;
    }

    outputBuffer = buffer;
    outputOffset = offset;
    return true;
}

bool CFDStreamlineStage::createPipeline() {
    VkDevice device = vulkanDevice.getDevice();

    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    for (uint32_t binding = 0; binding < bindings.size(); ++binding) {
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to create descriptor set layout." << std::endl;
        return false;
    }

    const uint32_t setCount = (maxFramesInFlight > 0) ? maxFramesInFlight : 1;

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(bindings.size()) * setCount;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to create descriptor pool." << std::endl;
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
        std::cerr << "[CFDStreamlineStage] Failed to allocate descriptor sets." << std::endl;
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(cfd::StreamlinePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to create pipeline layout." << std::endl;
        return false;
    }

    auto shaderCode = readFile("shaders/cfd_streamlines_comp.spv");
    if (shaderCode.empty()) {
        std::cerr << "[CFDStreamlineStage] Failed to read shaders/cfd_streamlines_comp.spv" << std::endl;
        return false;
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, shaderCode, shaderModule) != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to create shader module." << std::endl;
        return false;
    }

    VkComputePipelineCreateInfo computePipelineInfo{};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computePipelineInfo.stage.module = shaderModule;
    computePipelineInfo.stage.pName = "main";
    computePipelineInfo.layout = pipelineLayout;

    const VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (res != VK_SUCCESS) {
        std::cerr << "[CFDStreamlineStage] Failed to create compute pipeline." << std::endl;
        return false;
    }

    return true;
}

void CFDStreamlineStage::recordCompute(
    VkCommandBuffer cmd,
    uint32_t currentFrame,
    VkBuffer fluidSeedGridBuffer,
    VkDeviceSize fluidSeedGridOffset,
    VkBuffer velocityBuffer,
    VkDeviceSize velocityBufferOffset,
    uint32_t fluidNodeCount,
    float stepLength) {
    if (!ready ||
        descriptorSets.empty() ||
        fluidSeedGridBuffer == VK_NULL_HANDLE ||
        seedBuffer == VK_NULL_HANDLE ||
        seedToFluidNodeBuffer == VK_NULL_HANDLE ||
        velocityBuffer == VK_NULL_HANDLE ||
        outputBuffer == VK_NULL_HANDLE) {
        return;
    }

    const uint32_t frameIdx = currentFrame % static_cast<uint32_t>(descriptorSets.size());
    VkDescriptorSet currentSet = descriptorSets[frameIdx];

    const std::array<VkDescriptorBufferInfo, 5> bufferInfos = {{
        {fluidSeedGridBuffer, fluidSeedGridOffset, VK_WHOLE_SIZE},
        {seedToFluidNodeBuffer, seedToFluidNodeBufferOffset, VK_WHOLE_SIZE},
        {velocityBuffer, velocityBufferOffset, VK_WHOLE_SIZE},
        {seedBuffer, seedBufferOffset, VK_WHOLE_SIZE},
        {outputBuffer, outputOffset, VK_WHOLE_SIZE},
    }};

    std::array<VkWriteDescriptorSet, 5> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i) {
        writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, currentSet, i, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfos[i], nullptr};
    }
    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &currentSet, 0, nullptr);

    cfd::StreamlinePushConstants pushConstants{};
    pushConstants.gridMinCellSize = glm::vec4(gridMin, cellSize);
    pushConstants.gridDimMaxSegments = glm::ivec4(gridDim, static_cast<int>(MaxSegmentsPerLine));
    pushConstants.stepLengthPadding = glm::vec4(stepLength, 0.0f, 0.0f, 0.0f);
    pushConstants.counts = glm::uvec4(seedCount, seedToFluidNodeCount, fluidNodeCount, 0u);
    pushConstants.domainMin = glm::vec4(domainMin, 0.0f);
    pushConstants.domainMax = glm::vec4(domainMax, 0.0f);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    const uint32_t groupCount = (seedCount + 63u) / 64u;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    VkBufferMemoryBarrier exportBarrier{};
    exportBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    exportBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    exportBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    exportBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    exportBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    exportBarrier.buffer = outputBuffer;
    exportBarrier.offset = outputOffset;
    exportBarrier.size = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0,
        0, nullptr,
        1, &exportBarrier,
        0, nullptr);
}
