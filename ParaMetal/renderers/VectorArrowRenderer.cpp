#include "VectorArrowRenderer.hpp"

#include "util/Structs.hpp"
#include "util/file_utils.h"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/CommandBufferManager.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <unordered_set>

VectorArrowRenderer::VectorArrowRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uboManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uboManager),
      commandPool(commandPool) {
}

VectorArrowRenderer::~VectorArrowRenderer() {
    cleanup();
}

void VectorArrowRenderer::initialize(VkRenderPass renderPass, uint32_t maxFramesInFlight) {
    if (initialized) {
        cleanup();
    }

    if (!createArrowGeometry() ||
        !createDescriptorSetLayout() ||
        !createDescriptorPool(maxFramesInFlight) ||
        !createPipeline(renderPass)) {
        cleanup();
        return;
    }

    initialized = true;
}

bool VectorArrowRenderer::createArrowGeometry() {
    const std::array<glm::vec3, 10> vertices = {
        glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.78f, 0.08f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.78f, -0.08f, 0.0f),
        glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.78f, 0.0f, 0.08f),
        glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.78f, 0.0f, -0.08f),
    };

    vertexCount = static_cast<uint32_t>(vertices.size());
    const VkDeviceSize alignment = vulkanDevice.getPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment;

    uploadDeviceBuffer(
        memoryAllocator,
        commandPool,
        vertices.data(),
        sizeof(glm::vec3) * vertices.size(),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        alignment,
        vertexBuffer,
        vertexBufferOffset
    );

    return true;
}

bool VectorArrowRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "VectorArrowRenderer: Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool VectorArrowRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t maxVectorFields = 64;
    const uint32_t totalSets = maxFramesInFlight * maxVectorFields;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = totalSets * 2;  // 2 storage buffers (surface + gradient)
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = totalSets;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = totalSets;

    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "VectorArrowRenderer: Failed to create descriptor pool" << std::endl;
        return false;
    }

    return true;
}

bool VectorArrowRenderer::updateDescriptorSetVector(
    const VectorRenderBinding& vector,
    uint32_t maxFramesInFlight,
    std::vector<VkDescriptorSet>& targetSets) {
    if (targetSets.empty() || targetSets.size() != maxFramesInFlight) {
        targetSets.clear();
        targetSets.resize(maxFramesInFlight);

        std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = maxFramesInFlight;
        allocInfo.pSetLayouts = layouts.data();

        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, targetSets.data()) != VK_SUCCESS) {
            targetSets.clear();
            return false;
        }
    }

    VkDescriptorBufferInfo surfaceInfo{};
    surfaceInfo.buffer = vector.surfaceBuffer;
    surfaceInfo.offset = vector.surfaceBufferOffset;
    surfaceInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo gradientInfo{};
    gradientInfo.buffer = vector.gradientBuffer;
    gradientInfo.offset = vector.gradientBufferOffset;
    gradientInfo.range = VK_WHOLE_SIZE;

    for (uint32_t frame = 0; frame < maxFramesInFlight; ++frame) {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = uniformBufferManager.getUniformBuffers()[frame];
        uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frame];
        uboInfo.range = sizeof(UniformBufferObject);

        std::array<VkWriteDescriptorSet, 3> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = targetSets[frame];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &surfaceInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = targetSets[frame];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &gradientInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = targetSets[frame];
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[2].descriptorCount = 1;
        writes[2].pBufferInfo = &uboInfo;

        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return true;
}

void VectorArrowRenderer::updateDescriptors(
    const std::vector<VectorRenderBinding>& vectors,
    uint32_t maxFramesInFlight,
    bool forceReallocate) {
    if (!initialized) {
        return;
    }

    if (forceReallocate && descriptorPool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(vulkanDevice.getDevice(), descriptorPool, 0);
        vectorDescriptorSets.clear();
    }

    std::unordered_set<uint64_t> liveKeys;
    liveKeys.reserve(vectors.size());

    for (const VectorRenderBinding& vector : vectors) {
        if (vector.bindingKey == 0 || vector.gradientBuffer == VK_NULL_HANDLE || vector.sampleCount == 0) {
            continue;
        }

        liveKeys.insert(vector.bindingKey);
        auto& sets = vectorDescriptorSets[vector.bindingKey];
        if (!updateDescriptorSetVector(vector, maxFramesInFlight, sets)) {
            vectorDescriptorSets.erase(vector.bindingKey);
            std::cerr << "VectorArrowRenderer: Failed to allocate/update vector descriptor sets"
                      << " bindingKey=" << vector.bindingKey
                      << std::endl;
        }
    }

    for (auto it = vectorDescriptorSets.begin(); it != vectorDescriptorSets.end();) {
        if (liveKeys.find(it->first) == liveKeys.end()) {
            it = vectorDescriptorSets.erase(it);
        } else {
            ++it;
        }
    }
}

bool VectorArrowRenderer::createPipeline(VkRenderPass renderPass) {
    std::vector<char> vertShaderCode;
    std::vector<char> fragShaderCode;
    if (!readFile("shaders/vector_arrow_vert.spv", vertShaderCode) ||
        !readFile("shaders/vector_arrow_frag.spv", fragShaderCode)) {
        std::cerr << "VectorArrowRenderer: Failed to read shader files" << std::endl;
        return false;
    }

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        if (vertShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        }
        if (fragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        }
        std::cerr << "VectorArrowRenderer: Failed to create shader modules" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShaderModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShaderModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(glm::vec3);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.binding = 0;
    attributeDescription.location = 0;
    attributeDescription.format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDescription;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
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
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorAttachments[2] = {};
    colorAttachments[0].colorWriteMask = 0;
    colorAttachments[0].blendEnable = VK_FALSE;
    colorAttachments[1].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorAttachments[1].blendEnable = VK_TRUE;
    colorAttachments[1].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorAttachments[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorAttachments[1].colorBlendOp = VK_BLEND_OP_ADD;
    colorAttachments[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorAttachments[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorAttachments[1].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 2;
    colorBlending.pAttachments = colorAttachments;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "VectorArrowRenderer: Failed to create pipeline layout" << std::endl;
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 2;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "VectorArrowRenderer: Failed to create graphics pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return true;
}

void VectorArrowRenderer::render(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const std::vector<VectorRenderBinding>& vectors) const {
    if (!initialized ||
        pipeline == VK_NULL_HANDLE ||
        pipelineLayout == VK_NULL_HANDLE ||
        vertexBuffer == VK_NULL_HANDLE ||
        vectors.empty()) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkBuffer vertexBuffers[] = { vertexBuffer };
    VkDeviceSize vertexOffsets[] = { vertexBufferOffset };
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, vertexOffsets);

    for (const VectorRenderBinding& vector : vectors) {
        if (vector.bindingKey == 0 || vector.gradientBuffer == VK_NULL_HANDLE || vector.sampleCount == 0) {
            continue;
        }

        auto descriptorIt = vectorDescriptorSets.find(vector.bindingKey);
        if (descriptorIt == vectorDescriptorSets.end() || frameIndex >= descriptorIt->second.size()) {
            continue;
        }
        const VkDescriptorSet descriptorSet = descriptorIt->second[frameIndex];

        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);

        PushConstants pushConstants{};
        pushConstants.modelMatrix = vector.modelMatrix;
        pushConstants.scale = vector.scale;
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(PushConstants),
            &pushConstants);

        vkCmdDraw(commandBuffer, vertexCount, vector.sampleCount, 0, 0);
    }
}

void VectorArrowRenderer::cleanup() {
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
    freeBuffer(memoryAllocator, vertexBuffer, vertexBufferOffset);

    vertexCount = 0;
    vectorDescriptorSets.clear();
    initialized = false;
}
