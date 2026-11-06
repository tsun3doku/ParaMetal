#include "ProbeMarkerRenderer.hpp"

#include "ScreenTextRenderer.hpp"
#include "ui/UiPalette.hpp"
#include "util/Structs.hpp"
#include "util/file_utils.h"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <array>
#include <iostream>

ProbeMarkerRenderer::ProbeMarkerRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uniformBufferManager),
      commandPool(commandPool) {
}

ProbeMarkerRenderer::~ProbeMarkerRenderer() {
    cleanup();
}

void ProbeMarkerRenderer::initialize(
    VkRenderPass renderPass,
    uint32_t subpass,
    uint32_t maxFramesInFlight) {
    if (initialized) {
        cleanup();
    }

    if (!createTexture() ||
        !createDescriptorSetLayout() ||
        !createDescriptorPool(maxFramesInFlight) ||
        !createDescriptorSets(maxFramesInFlight) ||
        !createPipeline(renderPass, subpass)) {
        cleanup();
        return;
    }

    initialized = true;
}

bool ProbeMarkerRenderer::createTexture() {
    const std::string texturePath = "textures/icons/Marker/marker_b/128w/Artboard 1.png";
    uint32_t mipLevels = 1;
    if (createTextureImage(
            vulkanDevice,
            memoryAllocator,
            commandPool,
            texturePath,
            textureImage,
            textureImageMemory,
            mipLevels) != VK_SUCCESS) {
        std::cerr << "[ProbeMarkerRenderer] Failed to load marker texture: " << texturePath << std::endl;
        return false;
    }

    if (createTextureImageView(
            vulkanDevice,
            textureImage,
            textureImageView,
            mipLevels) != VK_SUCCESS) {
        std::cerr << "[ProbeMarkerRenderer] Failed to create marker texture image view" << std::endl;
        return false;
    }

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
        std::cerr << "[ProbeMarkerRenderer] Failed to create marker texture sampler" << std::endl;
        return false;
    }

    return true;
}

bool ProbeMarkerRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[ProbeMarkerRenderer] Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool ProbeMarkerRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = maxFramesInFlight;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = maxFramesInFlight;

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxFramesInFlight;

    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "[ProbeMarkerRenderer] Failed to create descriptor pool" << std::endl;
        return false;
    }

    return true;
}

bool ProbeMarkerRenderer::createDescriptorSets(uint32_t maxFramesInFlight) {
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = maxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(maxFramesInFlight);
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        descriptorSets.clear();
        std::cerr << "[ProbeMarkerRenderer] Failed to allocate descriptor sets" << std::endl;
        return false;
    }

    for (size_t i = 0; i < maxFramesInFlight; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBufferManager.getUniformBuffers()[i];
        bufferInfo.offset = uniformBufferManager.getUniformBufferOffsets()[i];
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptorSets[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    return true;
}

bool ProbeMarkerRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass) {
    std::vector<char> vertCode;
    std::vector<char> fragCode;
    if (!readFile("shaders/probe_marker_vert.spv", vertCode) ||
        !readFile("shaders/probe_marker_frag.spv", fragCode)) {
        std::cerr << "[ProbeMarkerRenderer] Failed to read shader files" << std::endl;
        return false;
    }

    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertCode, vertModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, fragCode, fragModule) != VK_SUCCESS) {
        if (vertModule != VK_NULL_HANDLE) vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        if (fragModule != VK_NULL_HANDLE) vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        std::cerr << "[ProbeMarkerRenderer] Failed to create shader modules" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    // No vertex input buffers needed - vertices generated in vertex shader via gl_VertexIndex
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = -2.0f;
    rasterizer.depthBiasSlopeFactor = -1.0f;
    rasterizer.depthBiasClamp = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        std::cerr << "[ProbeMarkerRenderer] Failed to create pipeline layout" << std::endl;
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
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

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        std::cerr << "[ProbeMarkerRenderer] Failed to create graphics pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);

    return true;
}

void ProbeMarkerRenderer::render(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    VkExtent2D extent,
    const glm::mat4& viewProj,
    const std::vector<RenderBinding>& bindings) {
    if (!initialized || bindings.empty() || frameIndex >= descriptorSets.size()) {
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        1,
        &descriptorSets[frameIndex],
        0,
        nullptr);

    const float vpW = static_cast<float>(extent.width);
    const float vpH = static_cast<float>(extent.height);

    for (const auto& binding : bindings) {
        PushConstants pc{};
        pc.worldPos = glm::vec4(binding.worldPosition, binding.scale);
        pc.colorTint = binding.color;
        pc.viewportSize = glm::vec2(vpW, vpH);
        pc.markerSize = glm::vec2(34.0f, 34.0f);
        pc.isSelected = binding.isSelected ? 1u : 0u;

        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(PushConstants),
            &pc);

        vkCmdDraw(commandBuffer, 6, 1, 0, 0);
    }

    if (screenTextRenderer && screenTextRenderer->isReady()) {
        renderLabels(commandBuffer, frameIndex, extent, viewProj, bindings);
    }
}

void ProbeMarkerRenderer::renderLabels(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    VkExtent2D extent,
    const glm::mat4& viewProj,
    const std::vector<RenderBinding>& bindings) {
    if (!screenTextRenderer || !screenTextRenderer->isReady() || bindings.empty()) {
        return;
    }

    const GlyphText& glyphText = screenTextRenderer->getGlyphText();
    const float planeHeight = glyphText.getPlaneHeight();
    if (planeHeight <= 0.0f) {
        return;
    }

    // 18px font size (bumped 2px for enhanced readability)
    const float emScale = 18.0f / planeHeight;

    std::vector<GlyphText::GlyphInstance> glyphs;
    glyphs.reserve(bindings.size() * 32);

    for (const auto& binding : bindings) {
        std::string labelText;
        if (binding.showName && !binding.probeName.empty()) {
            labelText = binding.probeName;
        }
        if (binding.showTemperature) {
            char tempBuf[32];
            std::snprintf(tempBuf, sizeof(tempBuf), "%.2f C", binding.currentTemperatureC);
            if (!labelText.empty()) {
                labelText += " ";
            }
            labelText += tempBuf;
        }

        if (labelText.empty()) {
            continue;
        }

        const glm::vec4 clipPos = viewProj * glm::vec4(binding.worldPosition, 1.0f);
        if (clipPos.w <= 0.001f) {
            continue;
        }

        const glm::vec3 ndc = glm::vec3(clipPos) / clipPos.w;
        if (ndc.z < 0.0f || ndc.z > 1.0f) {
            continue;
        }
        if (ndc.x < -1.1f || ndc.x > 1.1f || ndc.y < -1.1f || ndc.y > 1.1f) {
            continue;
        }

        const float screenX = (ndc.x * 0.5f + 0.5f) * static_cast<float>(extent.width);
        const float screenY = (ndc.y * 0.5f + 0.5f) * static_cast<float>(extent.height);

        // Position text label snug to the right of the pin marker bulb
        const float textStartX = screenX + 12.0f;
        const float textCenterY = screenY - 22.0f;

        const glm::vec4 shadowColor(0.02f, 0.02f, 0.02f, 0.90f);
        const glm::vec4 textColor = binding.isSelected
            ? ui::UiPalette::selectionHighlightSrgb()
            : glm::vec4(0.94f, 0.94f, 0.96f, 1.0f);   // Clean crisp white/light-grey

        // Two passes: 1. dark shadow at offset (1.0, 1.0), 2. crisp foreground
        for (int pass = 0; pass < 2; ++pass) {
            const float offsetX = (pass == 0) ? 1.0f : 0.0f;
            const float offsetY = (pass == 0) ? 1.0f : 0.0f;
            const glm::vec4& color = (pass == 0) ? shadowColor : textColor;

            float cursorX = textStartX + offsetX + glyphText.getCellOriginX() * emScale;
            const float baselineY = textCenterY + offsetY + glyphText.getCellOriginY() * emScale;

            for (char ch : labelText) {
                const GlyphText::CharInfo& info = glyphText.getCharInfo(ch);
                if (info.width > 0.0f && info.height > 0.0f) {
                    GlyphText::GlyphInstance glyph{};
                    glyph.centerPx = glm::vec2(
                        cursorX + 0.5f * (info.planeLeft + info.planeRight) * emScale,
                        baselineY + 0.5f * (info.planeTop + info.planeBottom) * emScale);
                    glyph.sizePx = glm::vec2(
                        (info.planeRight - info.planeLeft) * emScale,
                        (info.planeBottom - info.planeTop) * emScale);
                    glyph.charUV = glyphText.getCharUV(ch);
                    glyph.color = color;
                    if (glyph.charUV.z > 0.0f && glyph.charUV.w > 0.0f) {
                        glyphs.push_back(glyph);
                    }
                }
                cursorX += info.advanceEm * emScale;
            }
        }
    }

    if (!glyphs.empty()) {
        screenTextRenderer->draw(commandBuffer, frameIndex, extent, glyphs);
    }
}

void ProbeMarkerRenderer::cleanup() {
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
    if (textureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, textureSampler, nullptr);
        textureSampler = VK_NULL_HANDLE;
    }
    if (textureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, textureImageView, nullptr);
        textureImageView = VK_NULL_HANDLE;
    }
    if (textureImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, textureImage, nullptr);
        textureImage = VK_NULL_HANDLE;
    }
    if (textureImageMemory != VK_NULL_HANDLE) {
        memoryAllocator.freeImageMemory(textureImageMemory);
        textureImageMemory = VK_NULL_HANDLE;
    }

    descriptorSets.clear();
    initialized = false;
}
