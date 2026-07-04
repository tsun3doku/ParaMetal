#include "HeatSurfaceRenderer.hpp"

#include "heat/HeatGpuStructs.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "scene/Model.hpp"
#include "util/Structs.hpp"
#include "vulkan/VulkanImage.hpp"
#include "util/file_utils.h"

#include <array>
#include <glm/glm.hpp>
#include <iostream>
#include <unordered_set>

HeatSurfaceRenderer::HeatSurfaceRenderer(VulkanDevice& device, UniformBufferManager& uboManager)
    : vulkanDevice(device), uniformBufferManager(uboManager) {
}

HeatSurfaceRenderer::~HeatSurfaceRenderer() {
    cleanup();
}

void HeatSurfaceRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight) {
    if (initialized) {
        cleanup();
    }

    if (!createDescriptorPool(maxFramesInFlight) ||
        !createDescriptorSetLayout() ||
        !createPipeline(renderPass, subpass)) {
        cleanup();
        return;
    }

    initialized = true;
}

bool HeatSurfaceRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t maxRenderableHeatModels = 64;

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxRenderableHeatModels;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    poolSizes[1].descriptorCount = maxRenderableHeatModels * 10;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = maxRenderableHeatModels;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxRenderableHeatModels;

    descriptorPools.resize(maxFramesInFlight, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPools[i]) != VK_SUCCESS) {
            std::cerr << "HeatSurfaceRenderer: Failed to create descriptor pool " << i << std::endl;
            return false;
        }
    }

    return true;
}

bool HeatSurfaceRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 12> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    for (int i = 1; i <= 10; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    bindings[11].binding = 11;
    bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[11].descriptorCount = 1;
    bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "HeatSurfaceRenderer: Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool HeatSurfaceRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass) {
    std::vector<char> vertShaderCode;
    std::vector<char> geomShaderCode;
    std::vector<char> fragShaderCode;
    if (!readFile("shaders/intrinsic_supporting_vert.spv", vertShaderCode) ||
        !readFile("shaders/intrinsic_supporting_geom.spv", geomShaderCode) ||
        !readFile("shaders/heat_buffer_frag.spv", fragShaderCode)) {
        std::cerr << "HeatSurfaceRenderer: Failed to read shader files" << std::endl;
        return false;
    }

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    VkShaderModule geomShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, geomShaderCode, geomShaderModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        if (vertShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        }
        if (geomShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        }
        if (fragShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        }
        std::cerr << "HeatSurfaceRenderer: Failed to create shader modules" << std::endl;
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

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, geomShaderStageInfo, fragShaderStageInfo };

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
    rasterizer.depthBiasEnable = VK_TRUE;

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
    colorBlendAttachments[0].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = colorBlendAttachments;

    std::vector<VkDynamicState> dynamicStates = {
         VK_DYNAMIC_STATE_VIEWPORT,
         VK_DYNAMIC_STATE_SCISSOR,
         VK_DYNAMIC_STATE_DEPTH_BIAS
    };

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(heat::BufferPushConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "HeatSurfaceRenderer: Failed to create pipeline layout" << std::endl;
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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpass;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "HeatSurfaceRenderer: Failed to create graphics pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), geomShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return true;
}

void HeatSurfaceRenderer::drawModel(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, const SurfaceRenderBinding& binding) const {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    heat::BufferPushConstant pushConstants{};
    pushConstants.modelMatrix = binding.modelMatrix;
    pushConstants.sourceParams = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(heat::BufferPushConstant),
        &pushConstants);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &binding.vertexBuffer, &binding.vertexBufferOffset);
    vkCmdBindIndexBuffer(commandBuffer, binding.indexBuffer, binding.indexBufferOffset, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, binding.indexCount, 1, 0, 0, 0);
}

VkDescriptorSet HeatSurfaceRenderer::allocateDescriptorSet(VkDescriptorPool pool) {
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
            std::cerr << "HeatSurfaceRenderer: Descriptor pool exhaustion / failed to allocate descriptor set." << std::endl;
            loggedOnce = true;
        }
    }
    return set;
}

void HeatSurfaceRenderer::updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex, const SurfaceRenderBinding& binding) {
    std::array<VkWriteDescriptorSet, 12> descriptorWrites{};

    VkDescriptorBufferInfo uboBufferInfo{};
    uboBufferInfo.buffer = uniformBufferManager.getUniformBuffers()[frameIndex];
    uboBufferInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frameIndex];
    uboBufferInfo.range = sizeof(UniformBufferObject);

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = set;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &uboBufferInfo;

    for (int j = 0; j < 10; j++) {
        descriptorWrites[j + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[j + 1].dstSet = set;
        descriptorWrites[j + 1].dstBinding = 1 + j;
        descriptorWrites[j + 1].dstArrayElement = 0;
        descriptorWrites[j + 1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        descriptorWrites[j + 1].descriptorCount = 1;
        descriptorWrites[j + 1].pTexelBufferView = &binding.bufferViews[j];
    }

    VkDescriptorBufferInfo surfaceInfo{};
    surfaceInfo.buffer = binding.surfaceBuffer;
    surfaceInfo.offset = binding.surfaceBufferOffset;
    surfaceInfo.range = VK_WHOLE_SIZE;

    descriptorWrites[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[11].dstSet = set;
    descriptorWrites[11].dstBinding = 11;
    descriptorWrites[11].dstArrayElement = 0;
    descriptorWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[11].descriptorCount = 1;
    descriptorWrites[11].pBufferInfo = &surfaceInfo;

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), 12, descriptorWrites.data(), 0, nullptr);
}

void HeatSurfaceRenderer::render(VkCommandBuffer commandBuffer, uint32_t frameIndex, const std::vector<SurfaceRenderBinding>& surfaces) {
    if (!initialized || pipeline == VK_NULL_HANDLE || pipelineLayout == VK_NULL_HANDLE || frameIndex >= descriptorPools.size()) {
        return;
    }

    vkResetDescriptorPool(vulkanDevice.getDevice(), descriptorPools[frameIndex], 0);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (const SurfaceRenderBinding& surfaceBinding : surfaces) {
        if (surfaceBinding.runtimeModelId == 0) {
            continue;
        }

        VkDescriptorSet set = allocateDescriptorSet(descriptorPools[frameIndex]);
        if (set == VK_NULL_HANDLE) {
            continue;
        }

        updateDescriptorSet(set, frameIndex, surfaceBinding);
        drawModel(commandBuffer, set, surfaceBinding);
    }
}

void HeatSurfaceRenderer::cleanup() {
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

