#include "HeatPointRenderer.hpp"

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanImage.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "renderers/PointRenderer.hpp"
#include "util/Structs.hpp"
#include "util/file_utils.h"

#include <array>
#include <cstring>
#include <iostream>
#include <unordered_set>

HeatPointRenderer::HeatPointRenderer(VulkanDevice& device, UniformBufferManager& uniformBufferManager)
    : vulkanDevice(device), uniformBufferManager(uniformBufferManager) {
}

HeatPointRenderer::~HeatPointRenderer() {
    cleanup();
}

void HeatPointRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight) {
    if (initialized)
        return;

    if (!createDescriptorSetLayout() ||
        !createDescriptorPool(maxFramesInFlight) ||
        !createPipeline(renderPass, subpass)) {
        cleanup();
        return;
    }

    initialized = true;
}

bool HeatPointRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr,
        &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "HeatPointRenderer: Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool HeatPointRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t maxRenderablePointDomains = 64;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxRenderablePointDomains;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = maxRenderablePointDomains;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxRenderablePointDomains;

    descriptorPools.resize(maxFramesInFlight, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPools[i]) != VK_SUCCESS) {
            std::cerr << "HeatPointRenderer: Failed to create descriptor pool " << i << std::endl;
            return false;
        }
    }

    return true;
}

bool HeatPointRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass) {
    std::vector<char> vertCode;
    std::vector<char> fragCode;
    if (!readFile("shaders/heat_point_vert.spv", vertCode) ||
        !readFile("shaders/heat_point_frag.spv", fragCode)) {
        std::cerr << "HeatPointRenderer: Failed to read shader files" << std::endl;
        return false;
    }

    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertCode, vertModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, fragCode, fragModule) != VK_SUCCESS) {
        if (vertModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        }
        if (fragModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        }
        std::cerr << "HeatPointRenderer: Failed to create shader modules" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(PointRenderer::PointVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(PointRenderer::PointVertex, position);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
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
    depthStencil.stencilTestEnable = VK_TRUE;

    VkStencilOpState stencilOp{};
    stencilOp.passOp = VK_STENCIL_OP_KEEP;
    stencilOp.failOp = VK_STENCIL_OP_KEEP;
    stencilOp.depthFailOp = VK_STENCIL_OP_KEEP;
    stencilOp.compareOp = VK_COMPARE_OP_NOT_EQUAL;
    stencilOp.compareMask = 0xFF;
    stencilOp.writeMask = 0x00;
    stencilOp.reference = 0;
    depthStencil.front = stencilOp;
    depthStencil.back = stencilOp;

    VkPipelineColorBlendAttachmentState colorBlendAttachments[1] = {};
    colorBlendAttachments[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachments[0].blendEnable = VK_TRUE;
    colorBlendAttachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachments[0].colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachments[0].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = colorBlendAttachments;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4) + sizeof(float) * 2;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr,
        &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        std::cerr << "HeatPointRenderer: Failed to create pipeline layout" << std::endl;
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpass;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1,
        &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        std::cerr << "HeatPointRenderer: Failed to create graphics pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);

    return true;
}

VkDescriptorSet HeatPointRenderer::allocateDescriptorSet(VkDescriptorPool pool) {
    VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &set) != VK_SUCCESS) {
        static bool loggedOnce = false;
        if (!loggedOnce) {
            std::cerr << "HeatPointRenderer: Descriptor pool exhaustion / failed to allocate descriptor set." << std::endl;
            loggedOnce = true;
        }
    }
    return set;
}

void HeatPointRenderer::updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex, const PointRenderBinding& binding) {
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBufferManager.getUniformBuffers()[frameIndex];
    uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frameIndex];
    uboInfo.range = sizeof(UniformBufferObject);

    VkDescriptorBufferInfo tempInfo{};
    tempInfo.buffer = binding.tempBuffer;
    tempInfo.offset = binding.tempBufferOffset;
    tempInfo.range = VK_WHOLE_SIZE;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].dstArrayElement = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &uboInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].dstArrayElement = 0;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].descriptorCount = 1;
    writes[1].pBufferInfo = &tempInfo;

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void HeatPointRenderer::render(VkCommandBuffer commandBuffer, uint32_t frameIndex,
    const std::vector<PointRenderBinding>& bindings, VkExtent2D extent) {
    if (!initialized || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE || frameIndex >= descriptorPools.size())
        return;

    vkResetDescriptorPool(vulkanDevice.getDevice(), descriptorPools[frameIndex], 0);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (const auto& binding : bindings) {
        if (binding.pointCount == 0)
            continue;

        VkDescriptorSet set = allocateDescriptorSet(descriptorPools[frameIndex]);
        if (set == VK_NULL_HANDLE) {
            continue;
        }

        updateDescriptorSet(set, frameIndex, binding);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pipelineLayout, 0, 1, &set, 0, nullptr);

        struct {
            glm::mat4 model;
            float pointSize;
            float viewportHeight;
        } pushData = { binding.modelMatrix, binding.pointSize, static_cast<float>(extent.height) };

        vkCmdPushConstants(commandBuffer, pipelineLayout,
                          VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), &pushData);

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &binding.vertexBuffer, &binding.vertexBufferOffset);
        vkCmdDraw(commandBuffer, binding.pointCount, 1, 0, 0);
    }
}

void HeatPointRenderer::cleanup() {
    VkDevice device = vulkanDevice.getDevice();

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }

    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }

    for (auto pool : descriptorPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, pool, nullptr);
        }
    }
    descriptorPools.clear();

    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    initialized = false;
}
