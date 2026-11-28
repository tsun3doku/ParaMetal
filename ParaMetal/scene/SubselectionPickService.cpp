#include "SubselectionPickService.hpp"

#include "framegraph/VkFrameGraphRuntime.hpp"
#include "util/file_utils.h"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <array>
#include <cstring>
#include <iostream>

SubselectionPickService::SubselectionPickService(
    VulkanDevice& device,
    VkFrameGraphRuntime& runtime,
    framegraph::ResourceId resourceId,
    uint32_t maxFramesInFlight)
    : vulkanDevice(device),
      frameGraphRuntime(runtime),
      pickResourceId(resourceId) {
    if (!createFrameSlots(maxFramesInFlight) || !createComputeResources()) {
        std::cerr << "[SubselectionPickService] Initialization failed" << std::endl;
        cleanup();
    }
}

SubselectionPickService::~SubselectionPickService() {
    cleanup();
}

bool SubselectionPickService::createFrameSlots(uint32_t maxFramesInFlight) {
    if (maxFramesInFlight == 0) {
        return false;
    }
    frameSlots.resize(maxFramesInFlight);
    for (FrameSlot& slot : frameSlots) {
        if (!createFrameSlot(slot)) {
            return false;
        }
    }
    return true;
}

bool SubselectionPickService::createFrameSlot(FrameSlot& slot) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vulkanDevice.getDevice(), &info, nullptr, &slot.buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(vulkanDevice.getDevice(), slot.buffer, &requirements);

    uint32_t memoryTypeIndex = UINT32_MAX;
    if (!vulkanDevice.findMemoryType(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memoryTypeIndex)) {
        return false;
    }

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(vulkanDevice.getDevice(), &alloc, nullptr, &slot.memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(vulkanDevice.getDevice(), slot.buffer, slot.memory, 0) != VK_SUCCESS) {
        return false;
    }
    if (vkMapMemory(
            vulkanDevice.getDevice(),
            slot.memory,
            0,
            sizeof(uint32_t),
            0,
            &slot.mapped) != VK_SUCCESS) {
        return false;
    }

    VkBufferCreateInfo resultInfo{};
    resultInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    resultInfo.size = sizeof(uint32_t) * 4;
    resultInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    resultInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(
            vulkanDevice.getDevice(), &resultInfo, nullptr,
            &slot.computeResultBuffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements resultRequirements{};
    vkGetBufferMemoryRequirements(
        vulkanDevice.getDevice(), slot.computeResultBuffer, &resultRequirements);
    if (!vulkanDevice.findMemoryType(
            resultRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memoryTypeIndex)) {
        return false;
    }

    VkMemoryAllocateInfo resultAlloc = alloc;
    resultAlloc.allocationSize = resultRequirements.size;
    resultAlloc.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(
            vulkanDevice.getDevice(), &resultAlloc, nullptr,
            &slot.computeResultMemory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(
            vulkanDevice.getDevice(), slot.computeResultBuffer,
            slot.computeResultMemory, 0) != VK_SUCCESS) {
        return false;
    }
    return vkMapMemory(
        vulkanDevice.getDevice(),
        slot.computeResultMemory,
        0,
        sizeof(uint32_t) * 4,
        0,
        &slot.computeResultMapped) == VK_SUCCESS;
}

struct PickComputePushConstant {
    alignas(16) glm::mat4 mvp{1.0f};
    alignas(8) glm::vec2 cursor{0.0f};
    alignas(8) glm::vec2 viewport{0.0f};
    uint32_t radius = 0;
    uint32_t elementType = 0;
    uint32_t vertexStride = 11;
    uint32_t indexCount = 0;
};

bool SubselectionPickService::createComputeResources() {
    if (frameSlots.empty()) {
        return false;
    }

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = 0.0f;
    if (vkCreateSampler(
            vulkanDevice.getDevice(), &samplerInfo, nullptr,
            &pickSampler) != VK_SUCCESS) {
        return false;
    }

    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (uint32_t i = 1; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(
            vulkanDevice.getDevice(), &layoutInfo, nullptr,
            &computeSetLayout) != VK_SUCCESS) {
        return false;
    }

    const uint32_t frameCount = static_cast<uint32_t>(frameSlots.size());
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[0].descriptorCount = frameCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = frameCount * 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = frameCount;
    if (vkCreateDescriptorPool(
            vulkanDevice.getDevice(), &poolInfo, nullptr,
            &computeDescriptorPool) != VK_SUCCESS) {
        return false;
    }

    std::vector<VkDescriptorSetLayout> layouts(frameCount, computeSetLayout);
    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = computeDescriptorPool;
    setInfo.descriptorSetCount = frameCount;
    setInfo.pSetLayouts = layouts.data();
    computeDescriptorSets.resize(frameCount);
    if (vkAllocateDescriptorSets(
            vulkanDevice.getDevice(), &setInfo,
            computeDescriptorSets.data()) != VK_SUCCESS) {
        computeDescriptorSets.clear();
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(PickComputePushConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &computeSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(
            vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr,
            &computePipelineLayout) != VK_SUCCESS) {
        return false;
    }

    const std::vector<char> shaderCode =
        readFile("shaders/subselection_pick_comp.spv");
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (createShaderModule(
            vulkanDevice, shaderCode, shaderModule) != VK_SUCCESS) {
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = computePipelineLayout;

    const VkResult result = vkCreateComputePipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1,
        &pipelineInfo, nullptr, &computePipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), shaderModule, nullptr);
    return result == VK_SUCCESS;
}

void SubselectionPickService::setViewportExtent(uint32_t width, uint32_t height) {
    viewportWidth = width;
    viewportHeight = height;
}

void SubselectionPickService::queuePickRequest(const SubselectionPickRequest& request) {
    if (request.x < 0 || request.y < 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(queueMutex);
    pickQueue.push(request);
}

void SubselectionPickService::cancelThrough(
    uint64_t sessionId,
    uint64_t requestId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    std::queue<SubselectionPickRequest> filtered;
    while (!pickQueue.empty()) {
        const SubselectionPickRequest& request = pickQueue.front();
        if (request.sessionId != sessionId || request.requestId > requestId) {
            if (request.sessionId == sessionId) {
                filtered.push(request);
            }
        }
        pickQueue.pop();
    }
    pickQueue.swap(filtered);
}

const SubselectionPickRequest* SubselectionPickService::pendingRequest(
    uint32_t currentFrame) const {
    if (currentFrame >= frameSlots.size()) {
        return nullptr;
    }
    const FrameSlot& slot = frameSlots[currentFrame];
    return slot.requestPending ? &slot.request : nullptr;
}

void SubselectionPickService::cancelFrameRequest(uint32_t currentFrame) {
    if (currentFrame >= frameSlots.size()) {
        return;
    }
    FrameSlot& slot = frameSlots[currentFrame];
    if (slot.requestPending) {
        slot.cancelled = true;
        completeCancelledResult(currentFrame);
    }
}

struct GpuPickResult {
    uint32_t hit = 0;
    uint32_t padding = 0;
    uint32_t a = 0;
    uint32_t b = 0;
};

void SubselectionPickService::completeCancelledResult(uint32_t currentFrame) {
    if (currentFrame >= frameSlots.size()) {
        return;
    }
    FrameSlot& slot = frameSlots[currentFrame];
    SubselectionPickResult result{};
    result.sessionId = slot.request.sessionId;
    result.requestId = slot.request.requestId;
    result.elementType = slot.request.elementType;
    result.additive = slot.request.additive;
    result.cancelled = true;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        completedResults.push_back(result);
    }
    slot.requestPending = false;
    slot.copyPending = false;
    slot.computePending = false;
    slot.computeDescriptorPrepared = false;
    slot.cancelled = false;
}

bool SubselectionPickService::prepareComputePick(
    uint32_t currentFrame,
    const PickGeometryBinding& binding) {
    if (currentFrame >= frameSlots.size() ||
        currentFrame >= computeDescriptorSets.size()) {
        return false;
    }
    FrameSlot& slot = frameSlots[currentFrame];
    if (!slot.requestPending ||
        slot.cancelled ||
        slot.request.binding != SurfaceBinding::Source ||
        binding.vertexBuffer == VK_NULL_HANDLE ||
        binding.indexBuffer == VK_NULL_HANDLE ||
        binding.vertexRange == 0 ||
        binding.indexRange == 0 ||
        binding.indexCount == 0 ||
        binding.runtimeModelId != slot.request.runtimeModelId ||
        binding.geometryIdentity != slot.request.pickGeometryIdentity) {
        return false;
    }

    const auto& views = frameGraphRuntime.getResourceViews(pickResourceId);
    if (currentFrame >= views.size() || views[currentFrame] == VK_NULL_HANDLE) {
        return false;
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = views[currentFrame];
    imageInfo.sampler = pickSampler;

    VkDescriptorBufferInfo vertexInfo{};
    vertexInfo.buffer = binding.vertexBuffer;
    vertexInfo.offset = binding.vertexOffset;
    vertexInfo.range = binding.vertexRange;

    VkDescriptorBufferInfo indexInfo{};
    indexInfo.buffer = binding.indexBuffer;
    indexInfo.offset = binding.indexOffset;
    indexInfo.range = binding.indexRange;

    VkDescriptorBufferInfo resultInfo{};
    resultInfo.buffer = slot.computeResultBuffer;
    resultInfo.offset = 0;
    resultInfo.range = sizeof(GpuPickResult);

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = computeDescriptorSets[currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &imageInfo;
    for (uint32_t i = 1; i < writes.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = computeDescriptorSets[currentFrame];
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
    }
    writes[1].pBufferInfo = &vertexInfo;
    writes[2].pBufferInfo = &indexInfo;
    writes[3].pBufferInfo = &resultInfo;
    vkUpdateDescriptorSets(
        vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    slot.computeDescriptorPrepared = true;
    return true;
}
bool SubselectionPickService::recordComputePick(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    const PickGeometryBinding& binding) {
    if (!commandBuffer || currentFrame >= frameSlots.size()) {
        return false;
    }
    FrameSlot& slot = frameSlots[currentFrame];
    if (!slot.requestPending) {
        return false;
    }
    if (slot.cancelled ||
        slot.request.binding != SurfaceBinding::Source ||
        computePipeline == VK_NULL_HANDLE ||
        binding.vertexBuffer == VK_NULL_HANDLE ||
        binding.indexBuffer == VK_NULL_HANDLE ||
        binding.vertexRange == 0 ||
        binding.indexRange == 0 ||
        binding.indexCount == 0 ||
        binding.runtimeModelId != slot.request.runtimeModelId ||
        binding.geometryIdentity != slot.request.pickGeometryIdentity) {
        completeCancelledResult(currentFrame);
        return false;
    }

    const VkPhysicalDeviceLimits& limits =
        vulkanDevice.getPhysicalDeviceProperties().limits;
    if (binding.vertexRange > limits.maxStorageBufferRange ||
        binding.indexRange > limits.maxStorageBufferRange) {
        completeCancelledResult(currentFrame);
        return false;
    }

    if (!slot.computeDescriptorPrepared) {
        completeCancelledResult(currentFrame);
        return false;
    }

    const auto& images = frameGraphRuntime.getResourceImages(pickResourceId);
    if (currentFrame >= images.size() || images[currentFrame] == VK_NULL_HANDLE) {
        completeCancelledResult(currentFrame);
        return false;
    }

    VkImageMemoryBarrier imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = images[currentFrame];
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;
    imageBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &imageBarrier);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipelineLayout,
        0, 1,
        &computeDescriptorSets[currentFrame],
        0, nullptr);

    PickComputePushConstant push{};
    push.mvp = slot.request.viewProjection * slot.request.modelMatrix;
    push.cursor = glm::vec2(slot.request.x, slot.request.y);
    push.viewport = glm::vec2(slot.request.viewport);
    push.radius = slot.request.elementType == MeshElementType::Vertex ? 20u : 12u;
    push.elementType = static_cast<uint32_t>(slot.request.elementType);
    push.indexCount = binding.indexCount;
    vkCmdPushConstants(
        commandBuffer,
        computePipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(push),
        &push);
    vkCmdDispatch(commandBuffer, 1, 1, 1);

    VkBufferMemoryBarrier resultBarrier{};
    resultBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    resultBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    resultBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    resultBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    resultBarrier.buffer = slot.computeResultBuffer;
    resultBarrier.offset = 0;
    resultBarrier.size = sizeof(GpuPickResult);
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &resultBarrier, 0, nullptr);

    slot.requestPending = false;
    slot.computePending = true;
    return true;
}

void SubselectionPickService::processPickingRequests(uint32_t currentFrame) {
    if (currentFrame >= frameSlots.size()) {
        return;
    }

    FrameSlot& slot = frameSlots[currentFrame];
    if (slot.computePending && !slot.computeResultMapped) {
        completeCancelledResult(currentFrame);
    }
    if (slot.copyPending && !slot.mapped) {
        completeCancelledResult(currentFrame);
    }
    if (slot.computePending && slot.computeResultMapped) {
        const GpuPickResult* gpu =
            static_cast<const GpuPickResult*>(slot.computeResultMapped);
        SubselectionPickResult result{};
        result.sessionId = slot.request.sessionId;
        result.requestId = slot.request.requestId;
        result.elementType = slot.request.elementType;
        result.additive = slot.request.additive;
        result.hit = gpu->hit != 0;
        result.endpoints = glm::uvec2(gpu->a, gpu->b);
        result.elementId = result.endpoints.x;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            completedResults.push_back(result);
        }
        slot.computePending = false;
    }

    if (slot.copyPending && slot.mapped) {
        const uint32_t raw = *static_cast<uint32_t*>(slot.mapped);
        SubselectionPickResult result{};
        result.hit = raw != 0 && !slot.cancelled;
        result.cancelled = slot.cancelled;
        result.elementId = result.hit ? raw - 1u : 0u;
        result.endpoints.x = result.elementId;
        result.sessionId = slot.request.sessionId;
        result.requestId = slot.request.requestId;
        result.elementType = slot.request.elementType;
        result.additive = slot.request.additive;
        slot.cancelled = false;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            completedResults.push_back(result);
        }
        slot.copyPending = false;
    }

    if (slot.requestPending || slot.copyPending || slot.computePending) {
        return;
    }

    std::lock_guard<std::mutex> lock(queueMutex);
    if (pickQueue.empty()) {
        return;
    }
    slot.request = pickQueue.front();
    pickQueue.pop();
    slot.requestPending = true;
    slot.computeDescriptorPrepared = false;
    slot.cancelled = false;
}

void SubselectionPickService::recordReadback(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame) {
    if (!commandBuffer || currentFrame >= frameSlots.size()) {
        return;
    }

    FrameSlot& slot = frameSlots[currentFrame];
    if (!slot.requestPending ||
        (slot.request.elementType != MeshElementType::Face &&
         slot.request.binding != SurfaceBinding::Intrinsic)) {
        return;
    }

    if (slot.request.x < 0 || slot.request.y < 0 ||
        static_cast<uint32_t>(slot.request.x) >= viewportWidth ||
        static_cast<uint32_t>(slot.request.y) >= viewportHeight) {
        completeCancelledResult(currentFrame);
        return;
    }

    const auto& images = frameGraphRuntime.getResourceImages(pickResourceId);
    if (currentFrame >= images.size() || images[currentFrame] == VK_NULL_HANDLE ||
        slot.buffer == VK_NULL_HANDLE) {
        completeCancelledResult(currentFrame);
        return;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = images[currentFrame];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {slot.request.x, slot.request.y, 0};
    region.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(
        commandBuffer,
        images[currentFrame],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        slot.buffer,
        1,
        &region);

    slot.requestPending = false;
    slot.copyPending = true;
}

std::optional<SubselectionPickResult> SubselectionPickService::takeResult() {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (completedResults.empty()) {
        return std::nullopt;
    }
    const SubselectionPickResult result = completedResults.front();
    completedResults.pop_front();
    return result;
}

void SubselectionPickService::cleanup() {
    for (FrameSlot& slot : frameSlots) {
        if (slot.mapped) {
            vkUnmapMemory(vulkanDevice.getDevice(), slot.memory);
            slot.mapped = nullptr;
        }
        if (slot.computeResultMapped) {
            vkUnmapMemory(vulkanDevice.getDevice(), slot.computeResultMemory);
            slot.computeResultMapped = nullptr;
        }
        if (slot.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(vulkanDevice.getDevice(), slot.buffer, nullptr);
            slot.buffer = VK_NULL_HANDLE;
        }
        if (slot.computeResultBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(
                vulkanDevice.getDevice(), slot.computeResultBuffer, nullptr);
            slot.computeResultBuffer = VK_NULL_HANDLE;
        }
        if (slot.memory != VK_NULL_HANDLE) {
            vkFreeMemory(vulkanDevice.getDevice(), slot.memory, nullptr);
            slot.memory = VK_NULL_HANDLE;
        }
        if (slot.computeResultMemory != VK_NULL_HANDLE) {
            vkFreeMemory(
                vulkanDevice.getDevice(), slot.computeResultMemory, nullptr);
            slot.computeResultMemory = VK_NULL_HANDLE;
        }
    }
    frameSlots.clear();
    computeDescriptorSets.clear();
    if (computePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), computePipeline, nullptr);
        computePipeline = VK_NULL_HANDLE;
    }
    if (computePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            vulkanDevice.getDevice(), computePipelineLayout, nullptr);
        computePipelineLayout = VK_NULL_HANDLE;
    }
    if (computeDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            vulkanDevice.getDevice(), computeDescriptorPool, nullptr);
        computeDescriptorPool = VK_NULL_HANDLE;
    }
    if (computeSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            vulkanDevice.getDevice(), computeSetLayout, nullptr);
        computeSetLayout = VK_NULL_HANDLE;
    }
    if (pickSampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            vulkanDevice.getDevice(), pickSampler, nullptr);
        pickSampler = VK_NULL_HANDLE;
    }
}
