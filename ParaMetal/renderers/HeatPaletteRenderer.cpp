#include "HeatPaletteRenderer.hpp"

#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "util/file_utils.h"
#include "libs/stb/stb_image.h"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>

HeatPaletteRenderer::HeatPaletteRenderer(VulkanDevice& vulkanDevice, MemoryAllocator& allocator, CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      allocator(allocator),
      commandPool(commandPool) {
}

void HeatPaletteRenderer::initialize(VkRenderPass renderPass, uint32_t subpassIndex, uint32_t maxFramesInFlight) {
    this->maxFramesInFlight = maxFramesInFlight;
    if (!glyphText.load()) {
        std::cerr << "[HeatPaletteRenderer] Failed to load glyph text" << std::endl;
        return;
    }
    createBarPipeline(renderPass, subpassIndex);
    createQuadVertexBuffer();
    createTextInstanceBuffers(maxFramesInFlight);
    createFontAtlas();
    createTextDescriptorPool(maxFramesInFlight);
    createTextDescriptorSetLayout();
    createTextPipeline(renderPass, subpassIndex);
    createTextDescriptorSets(maxFramesInFlight);
}

HeatPaletteRenderer::~HeatPaletteRenderer() {
    cleanup();
}

void HeatPaletteRenderer::setVisible(bool visible) {
    this->visible = visible;
}

void HeatPaletteRenderer::setRange(float minTemp, float maxTemp) {
    if (this->minTemp == minTemp && this->maxTemp == maxTemp) {
        return;
    }
    this->minTemp = minTemp;
    this->maxTemp = maxTemp;
    glyphInstancesDirty = true;
}

void HeatPaletteRenderer::createBarPipeline(VkRenderPass renderPass, uint32_t subpassIndex) {
    const std::vector<char> vertCode = readFile("shaders/heat_palette_bar_vert.spv");
    const std::vector<char> fragCode = readFile("shaders/heat_palette_bar_frag.spv");

    VkShaderModule vertShader = createShaderModule(vulkanDevice, vertCode);
    VkShaderModule fragShader = createShaderModule(vulkanDevice, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 0;
    vertexInput.pVertexBindingDescriptions = nullptr;
    vertexInput.vertexAttributeDescriptionCount = 0;
    vertexInput.pVertexAttributeDescriptions = nullptr;

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
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendAttachmentState blendAttachments[2] = {};
    blendAttachments[0].colorWriteMask = 0;
    blendAttachments[0].blendEnable = VK_FALSE;
    blendAttachments[1].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachments[1].blendEnable = VK_TRUE;
    blendAttachments[1].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachments[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachments[1].colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachments[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachments[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachments[1].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 2;
    colorBlend.pAttachments = blendAttachments;

    std::array<VkDynamicState, 2> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(BarPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &barPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
        std::cerr << "[HeatPaletteRenderer] Failed to create bar pipeline layout" << std::endl;
        return;
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
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = barPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpassIndex;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &barPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), barPipelineLayout, nullptr);
        barPipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
        std::cerr << "[HeatPaletteRenderer] Failed to create bar pipeline" << std::endl;
        return;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
}

void HeatPaletteRenderer::createTextPipeline(VkRenderPass renderPass, uint32_t subpassIndex) {
    const std::vector<char> vertCode = readFile("shaders/timing_overlay_vert.spv");
    const std::vector<char> fragCode = readFile("shaders/timing_overlay_frag.spv");

    VkShaderModule vertShader = createShaderModule(vulkanDevice, vertCode);
    VkShaderModule fragShader = createShaderModule(vulkanDevice, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShader;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShader;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStage, fragStage };

    VkVertexInputBindingDescription vertexBinding{};
    vertexBinding.binding = 0;
    vertexBinding.stride = sizeof(glm::vec2) * 2;
    vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputBindingDescription instanceBinding{};
    instanceBinding.binding = 1;
    instanceBinding.stride = sizeof(GlyphText::GlyphInstance);
    instanceBinding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputBindingDescription, 2> bindings = { vertexBinding, instanceBinding };

    std::array<VkVertexInputAttributeDescription, 6> attrs{};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 };
    attrs[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT, sizeof(glm::vec2) };
    attrs[2] = { 2, 1, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphText::GlyphInstance, centerPx)) };
    attrs[3] = { 3, 1, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphText::GlyphInstance, sizePx)) };
    attrs[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphText::GlyphInstance, charUV)) };
    attrs[5] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(GlyphText::GlyphInstance, color)) };

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
    vertexInput.pVertexBindingDescriptions = bindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    VkPipelineColorBlendAttachmentState blendAttachments[2] = {};
    blendAttachments[0].colorWriteMask = 0;
    blendAttachments[0].blendEnable = VK_FALSE;
    blendAttachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachments[1].blendEnable = VK_TRUE;
    blendAttachments[1].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachments[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachments[1].colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachments[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachments[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachments[1].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 2;
    colorBlend.pAttachments = blendAttachments;

    std::array<VkDynamicState, 2> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::vec2);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &textDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &textPipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
        std::cerr << "[HeatPaletteRenderer] Failed to create text pipeline layout" << std::endl;
        return;
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
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = textPipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpassIndex;

    if (vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &textPipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), textPipelineLayout, nullptr);
        textPipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
        std::cerr << "[HeatPaletteRenderer] Failed to create text pipeline" << std::endl;
        return;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShader, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShader, nullptr);
}

void HeatPaletteRenderer::createQuadVertexBuffer() {
    const std::array<glm::vec2, 4> positions = {
        glm::vec2(-0.5f, -0.5f),
        glm::vec2( 0.5f, -0.5f),
        glm::vec2( 0.5f,  0.5f),
        glm::vec2(-0.5f,  0.5f),
    };
    const std::array<glm::vec2, 4> texCoords = {
        glm::vec2(0.0f, 1.0f),
        glm::vec2(1.0f, 1.0f),
        glm::vec2(1.0f, 0.0f),
        glm::vec2(0.0f, 0.0f),
    };

    struct Vertex { glm::vec2 pos; glm::vec2 uv; };
    const std::array<Vertex, 4> vertices = {
        Vertex{positions[0], texCoords[0]},
        Vertex{positions[1], texCoords[1]},
        Vertex{positions[2], texCoords[2]},
        Vertex{positions[3], texCoords[3]},
    };

    const VkDeviceSize size = sizeof(Vertex) * vertices.size();

    auto [buffer, offset] = allocator.allocate(
        size,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (buffer == VK_NULL_HANDLE) {
        std::cerr << "[HeatPaletteRenderer] Failed to allocate quad vertex buffer" << std::endl;
        return;
    }

    quadVertexBuffer = buffer;
    quadVertexBufferOffset = offset;

    void* mapped = allocator.getMappedPointer(buffer, offset);
    if (mapped) {
        std::memcpy(mapped, vertices.data(), static_cast<size_t>(size));
    }
}

void HeatPaletteRenderer::createTextInstanceBuffers(uint32_t maxFramesInFlight) {
    const VkDeviceSize size = sizeof(GlyphText::GlyphInstance) * maxGlyphCapacity;

    textInstanceBuffers.resize(maxFramesInFlight, VK_NULL_HANDLE);
    textInstanceBufferOffsets.resize(maxFramesInFlight, 0);
    textInstanceBuffersMapped.resize(maxFramesInFlight, nullptr);

    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        auto [buffer, offset] = allocator.allocate(
            size,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (buffer == VK_NULL_HANDLE) {
            std::cerr << "[HeatPaletteRenderer] Failed to allocate instance buffer [" << i << "]" << std::endl;
            return;
        }

        textInstanceBuffers[i] = buffer;
        textInstanceBufferOffsets[i] = offset;
        textInstanceBuffersMapped[i] = allocator.getMappedPointer(buffer, offset);
    }
}

void HeatPaletteRenderer::createFontAtlas() {
    int width = 0;
    int height = 0;
    const std::string& atlasPath = glyphText.getAtlasTexturePath();
    stbi_uc* pixels = stbi_load(atlasPath.c_str(), &width, &height, nullptr, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "[HeatPaletteRenderer] Failed to load font atlas: " << atlasPath << std::endl;
        return;
    }

    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

    VkBufferCreateInfo stagingInfo{};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(vulkanDevice.getDevice(), &stagingInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
        stbi_image_free(pixels);
        std::cerr << "[HeatPaletteRenderer] Failed to create staging buffer" << std::endl;
        return;
    }

    VkMemoryRequirements stagingReq{};
    vkGetBufferMemoryRequirements(vulkanDevice.getDevice(), stagingBuffer, &stagingReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = stagingReq.size;
    allocInfo.memoryTypeIndex = vulkanDevice.findMemoryType(
        stagingReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(vulkanDevice.getDevice(), &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(vulkanDevice.getDevice(), stagingBuffer, nullptr);
        stbi_image_free(pixels);
        std::cerr << "[HeatPaletteRenderer] Failed to allocate staging memory" << std::endl;
        return;
    }

    vkBindBufferMemory(vulkanDevice.getDevice(), stagingBuffer, stagingMemory, 0);

    void* mapped = nullptr;
    vkMapMemory(vulkanDevice.getDevice(), stagingMemory, 0, imageSize, 0, &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(vulkanDevice.getDevice(), stagingMemory);
    stbi_image_free(pixels);

    createImage(
        vulkanDevice,
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        fontAtlasImage,
        fontAtlasMemory,
        VK_SAMPLE_COUNT_1_BIT,
        1);

    transitionImageLayout(commandPool, fontAtlasImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
    commandPool.copyBufferToImage(stagingBuffer, fontAtlasImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    transitionImageLayout(commandPool, fontAtlasImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    fontAtlasView = createImageView(vulkanDevice, fontAtlasImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 1);

    vkDestroyBuffer(vulkanDevice.getDevice(), stagingBuffer, nullptr);
    vkFreeMemory(vulkanDevice.getDevice(), stagingMemory, nullptr);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    samplerInfo.anisotropyEnable = VK_FALSE;

    if (vkCreateSampler(vulkanDevice.getDevice(), &samplerInfo, nullptr, &fontSampler) != VK_SUCCESS) {
        std::cerr << "[HeatPaletteRenderer] Failed to create font sampler" << std::endl;
        return;
    }
}

void HeatPaletteRenderer::createTextDescriptorPool(uint32_t maxFramesInFlight) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = maxFramesInFlight;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = maxFramesInFlight;

    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &textDescriptorPool) != VK_SUCCESS) {
        std::cerr << "[HeatPaletteRenderer] Failed to create descriptor pool" << std::endl;
        return;
    }
}

void HeatPaletteRenderer::createTextDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding = 0;
    samplerBinding.descriptorCount = 1;
    samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &samplerBinding;

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &textDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "[HeatPaletteRenderer] Failed to create descriptor set layout" << std::endl;
        return;
    }
}

void HeatPaletteRenderer::createTextDescriptorSets(uint32_t maxFramesInFlight) {
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, textDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = textDescriptorPool;
    allocInfo.descriptorSetCount = maxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();

    textDescriptorSets.resize(maxFramesInFlight, VK_NULL_HANDLE);
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, textDescriptorSets.data()) != VK_SUCCESS) {
        std::cerr << "[HeatPaletteRenderer] Failed to allocate descriptor sets" << std::endl;
        return;
    }

    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = fontAtlasView;
        imageInfo.sampler = fontSampler;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = textDescriptorSets[i];
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), 1, &write, 0, nullptr);
    }
}

void HeatPaletteRenderer::buildGlyphInstances(VkExtent2D extent) {
    glyphInstances.clear();

    const float scale = 18.0f / 64.0f;
    const float advanceFactor = 0.4f;
    const glm::vec4 labelColor(0.9f, 0.9f, 0.9f, 1.0f);

    const float barWidth = 16.0f;
    const float barHeight = 320.0f;
    const float marginLeft = 20.0f;
    const float marginBottom = 20.0f;
    const float labelOffsetX = 8.0f;

    float barLeft = marginLeft;
    float barBottom = marginBottom;

    auto addLabel = [&](float temp, float tickY) {
        std::ostringstream oss;
        oss << static_cast<int>(temp) << "C";
        std::string label = oss.str();

        float labelMinY = 1e9f;
        float labelMaxY = -1e9f;
        std::vector<std::pair<char, GlyphText::CharInfo>> labelGlyphs;
        labelGlyphs.reserve(label.size());

        for (char c : label) {
            const GlyphText::CharInfo& info = glyphText.getCharInfo(c);
            labelGlyphs.push_back({c, info});
            if (info.width > 0.0f && info.height > 0.0f) {
                float gy = (info.yoffset * scale) + (0.5f * info.height * scale);
                float halfH = 0.5f * info.height * scale;
                labelMinY = std::min(labelMinY, gy - halfH);
                labelMaxY = std::max(labelMaxY, gy + halfH);
            }
        }

        float labelHeight = labelMaxY - labelMinY;
        float cursorX = barLeft + barWidth + labelOffsetX;
        float cursorY = tickY - (labelHeight * 0.5f) - labelMinY;

        for (const auto& [c, info] : labelGlyphs) {
            if (info.width > 0.0f && info.height > 0.0f) {
                GlyphText::GlyphInstance glyph{};
                glyph.centerPx = glm::vec2(
                    cursorX + (info.xoffset * scale) + (0.5f * info.width * scale),
                    cursorY + (info.yoffset * scale) + (0.5f * info.height * scale));
                glyph.sizePx = glm::vec2(info.width * scale, info.height * scale);
                glyph.charUV = glyphText.getCharUV(c);
                glyph.color = labelColor;

                if (glyph.charUV.z > 0.0f && glyph.charUV.w > 0.0f) {
                    glyphInstances.push_back(glyph);
                    if (glyphInstances.size() >= maxGlyphCapacity) return;
                }
            }
            cursorX += info.xadvance * scale * advanceFactor;
        }
    };

    addLabel(minTemp, barBottom);
    addLabel(maxTemp, barBottom + barHeight);

    glyphCount = static_cast<uint32_t>(glyphInstances.size());
    glyphInstancesDirty = false;

    for (void* mapped : textInstanceBuffersMapped) {
        if (!mapped) continue;
        if (!glyphInstances.empty()) {
            std::memcpy(mapped, glyphInstances.data(), sizeof(GlyphText::GlyphInstance) * glyphInstances.size());
        }
    }
}

void HeatPaletteRenderer::render(VkCommandBuffer commandBuffer, uint32_t currentFrame, VkExtent2D extent) {
    if (!visible) return;

    if (barPipeline == VK_NULL_HANDLE || textPipeline == VK_NULL_HANDLE) return;

    if (glyphInstancesDirty || viewportExtent.width != extent.width || viewportExtent.height != extent.height) {
        viewportExtent = extent;
        buildGlyphInstances(extent);
    }

    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = extent;

    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // Bar pass
    BarPushConstants barPush{};
    barPush.barRect = glm::vec4(20.0f, 20.0f, 16.0f, 320.0f);
    barPush.viewportSize = glm::vec2(static_cast<float>(extent.width), static_cast<float>(extent.height));
    barPush.minTemp = minTemp;
    barPush.maxTemp = maxTemp;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, barPipeline);
    vkCmdPushConstants(commandBuffer, barPipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(BarPushConstants), &barPush);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    // Text pass
    if (glyphCount > 0 && currentFrame < textDescriptorSets.size() && currentFrame < textInstanceBuffers.size()) {
        const glm::vec2 viewportSize(static_cast<float>(extent.width), static_cast<float>(extent.height));

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, textPipelineLayout, 0, 1,
            &textDescriptorSets[currentFrame], 0, nullptr);
        vkCmdPushConstants(commandBuffer, textPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
            sizeof(glm::vec2), &viewportSize);

        VkBuffer buffers[2] = { quadVertexBuffer, textInstanceBuffers[currentFrame] };
        VkDeviceSize offsets[2] = { quadVertexBufferOffset, textInstanceBufferOffsets[currentFrame] };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, buffers, offsets);
        vkCmdDraw(commandBuffer, 4, glyphCount, 0, 0);
    }
}

void HeatPaletteRenderer::cleanup() {
    VkDevice device = vulkanDevice.getDevice();

    if (barPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, barPipeline, nullptr);
    if (barPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, barPipelineLayout, nullptr);
    if (textPipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, textPipeline, nullptr);
    if (textPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, textPipelineLayout, nullptr);
    if (textDescriptorSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, textDescriptorSetLayout, nullptr);
    if (textDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, textDescriptorPool, nullptr);
    if (fontSampler != VK_NULL_HANDLE) vkDestroySampler(device, fontSampler, nullptr);
    if (fontAtlasView != VK_NULL_HANDLE) vkDestroyImageView(device, fontAtlasView, nullptr);
    if (fontAtlasImage != VK_NULL_HANDLE) vkDestroyImage(device, fontAtlasImage, nullptr);
    if (fontAtlasMemory != VK_NULL_HANDLE) vkFreeMemory(device, fontAtlasMemory, nullptr);

    for (size_t i = 0; i < textInstanceBuffers.size(); ++i) {
        if (textInstanceBuffers[i] != VK_NULL_HANDLE) {
            allocator.free(textInstanceBuffers[i], textInstanceBufferOffsets[i]);
        }
    }

    if (quadVertexBuffer != VK_NULL_HANDLE) {
        allocator.free(quadVertexBuffer, quadVertexBufferOffset);
        quadVertexBuffer = VK_NULL_HANDLE;
    }
    quadVertexBufferOffset = 0;

    textInstanceBuffers.clear();
    textInstanceBufferOffsets.clear();
    textInstanceBuffersMapped.clear();
    textDescriptorSets.clear();

    barPipeline = VK_NULL_HANDLE;
    barPipelineLayout = VK_NULL_HANDLE;
    textPipeline = VK_NULL_HANDLE;
    textPipelineLayout = VK_NULL_HANDLE;
    textDescriptorSetLayout = VK_NULL_HANDLE;
    textDescriptorPool = VK_NULL_HANDLE;
    fontSampler = VK_NULL_HANDLE;
    fontAtlasView = VK_NULL_HANDLE;
    fontAtlasImage = VK_NULL_HANDLE;
    fontAtlasMemory = VK_NULL_HANDLE;
}
