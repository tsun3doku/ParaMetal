#include "HeatVolumeRenderer.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <array>

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "util/file_utils.h"
#include "util/Structs.hpp"

namespace render {

HeatVolumeRenderer::HeatVolumeRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uniformBufferManager),
      commandPool(commandPool) {
}

HeatVolumeRenderer::~HeatVolumeRenderer() {
    cleanup();
}

bool HeatVolumeRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t updatedMaxFramesInFlight) {
    if (initialized) {
        cleanup();
    }

    maxFramesInFlight = updatedMaxFramesInFlight;

    if (!createPaletteTextures()) {
        std::cerr << "[HeatVolumeRenderer] Failed to create palette textures." << std::endl;
        return false;
    }

    if (!createDescriptorSetLayout()) {
        std::cerr << "[HeatVolumeRenderer] Failed to create descriptor set layout." << std::endl;
        destroyPaletteTextures();
        return false;
    }

    if (!createDescriptorPool(maxFramesInFlight)) {
        std::cerr << "[HeatVolumeRenderer] Failed to create descriptor pool." << std::endl;
        cleanup();
        return false;
    }

    if (!createDescriptorSets(maxFramesInFlight)) {
        std::cerr << "[HeatVolumeRenderer] Failed to create descriptor sets." << std::endl;
        cleanup();
        return false;
    }

    if (!createPipeline(renderPass, subpass)) {
        std::cerr << "[HeatVolumeRenderer] Failed to create graphics pipeline." << std::endl;
        cleanup();
        return false;
    }

    initialized = true;
    return true;
}

void HeatVolumeRenderer::cleanup() {
    VkDevice device = vulkanDevice.getDevice();

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    for (VkDescriptorPool pool : descriptorPools) {
        if (pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, pool, nullptr);
        }
    }
    descriptorPools.clear();
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    descriptorSets.clear();

    destroyPaletteTextures();

    volumeImageView = VK_NULL_HANDLE;
    volumeSampler = VK_NULL_HANDLE;
    hasDomain = false;
    worldToDomain = glm::mat4(1.0f);
    initialized = false;
}

bool HeatVolumeRenderer::createPaletteTextures() {
    if (createTextureImage16(vulkanDevice, memoryAllocator, commandPool, "textures/palettes/inferno.png", infernoImage, infernoMemory) != VK_SUCCESS ||
        createTextureImage16(vulkanDevice, memoryAllocator, commandPool, "textures/palettes/viridis.png", viridisImage, viridisMemory) != VK_SUCCESS ||
        createTextureImage16(vulkanDevice, memoryAllocator, commandPool, "textures/palettes/inferno2.png", inferno2Image, inferno2Memory) != VK_SUCCESS ||
        createTextureImage16(vulkanDevice, memoryAllocator, commandPool, "textures/palettes/parula.png", parulaImage, parulaMemory) != VK_SUCCESS ||
        createImageView(vulkanDevice, infernoImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, infernoView) != VK_SUCCESS ||
        createImageView(vulkanDevice, viridisImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, viridisView) != VK_SUCCESS ||
        createImageView(vulkanDevice, inferno2Image, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, inferno2View) != VK_SUCCESS ||
        createImageView(vulkanDevice, parulaImage, VK_FORMAT_R16G16B16A16_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, parulaView) != VK_SUCCESS ||
        createTextureSampler(vulkanDevice, paletteSampler) != VK_SUCCESS) {
        destroyPaletteTextures();
        return false;
    }
    return true;
}

void HeatVolumeRenderer::destroyPaletteTextures() {
    VkDevice device = vulkanDevice.getDevice();
    if (paletteSampler != VK_NULL_HANDLE) vkDestroySampler(device, paletteSampler, nullptr);
    if (infernoView != VK_NULL_HANDLE) vkDestroyImageView(device, infernoView, nullptr);
    if (viridisView != VK_NULL_HANDLE) vkDestroyImageView(device, viridisView, nullptr);
    if (inferno2View != VK_NULL_HANDLE) vkDestroyImageView(device, inferno2View, nullptr);
    if (parulaView != VK_NULL_HANDLE) vkDestroyImageView(device, parulaView, nullptr);
    if (infernoImage != VK_NULL_HANDLE) vkDestroyImage(device, infernoImage, nullptr);
    if (viridisImage != VK_NULL_HANDLE) vkDestroyImage(device, viridisImage, nullptr);
    if (inferno2Image != VK_NULL_HANDLE) vkDestroyImage(device, inferno2Image, nullptr);
    if (parulaImage != VK_NULL_HANDLE) vkDestroyImage(device, parulaImage, nullptr);
    if (infernoMemory != VK_NULL_HANDLE) memoryAllocator.freeImageMemory(infernoMemory);
    if (viridisMemory != VK_NULL_HANDLE) memoryAllocator.freeImageMemory(viridisMemory);
    if (inferno2Memory != VK_NULL_HANDLE) memoryAllocator.freeImageMemory(inferno2Memory);
    if (parulaMemory != VK_NULL_HANDLE) memoryAllocator.freeImageMemory(parulaMemory);
    paletteSampler = VK_NULL_HANDLE;
    infernoView = viridisView = inferno2View = parulaView = VK_NULL_HANDLE;
    infernoImage = viridisImage = inferno2Image = parulaImage = VK_NULL_HANDLE;
    infernoMemory = viridisMemory = inferno2Memory = parulaMemory = VK_NULL_HANDLE;
}

bool HeatVolumeRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};

    // Binding 0: Camera UBO
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Binding 1: 3D Fluid Temperature Volume
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    // Bindings 2-5: Palettes
    for (uint32_t i = 2; i <= 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    return vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) == VK_SUCCESS;
}

bool HeatVolumeRenderer::createDescriptorPool(uint32_t count) {
    const uint32_t setCount = std::max(count, 256u);
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = setCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = setCount * 5;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = setCount;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        return false;
    }
    descriptorPools.push_back(pool);
    return true;
}

VkDescriptorSet HeatVolumeRenderer::allocateDescriptorSet() {
    if (descriptorPools.empty() && !createDescriptorPool(256)) {
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPools.back();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &set) != VK_SUCCESS) {
        if (!createDescriptorPool(256)) {
            return VK_NULL_HANDLE;
        }
        allocInfo.descriptorPool = descriptorPools.back();
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &set) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
    }
    return set;
}

bool HeatVolumeRenderer::createDescriptorSets(uint32_t count) {
    if (descriptorPools.empty() && !createDescriptorPool(count)) {
        return false;
    }

    descriptorSets.assign(count, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < count; ++i) {
        descriptorSets[i] = allocateDescriptorSet();
        if (descriptorSets[i] == VK_NULL_HANDLE) {
            return false;
        }
    }

    updateStaticDescriptorSets();
    return true;
}

void HeatVolumeRenderer::writeDescriptorSet(
    VkDescriptorSet set,
    uint32_t frameIndex,
    bool includeVolume) {
    if (set == VK_NULL_HANDLE || descriptorSetLayout == VK_NULL_HANDLE) {
        return;
    }

    std::array<VkWriteDescriptorSet, 6> writes{};
    uint32_t writeCount = 0;

    // 0: Camera UBO
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBufferManager.getUniformBuffers()[frameIndex];
    uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frameIndex];
    uboInfo.range = sizeof(UniformBufferObject);

    writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[writeCount].dstSet = set;
    writes[writeCount].dstBinding = 0;
    writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[writeCount].descriptorCount = 1;
    writes[writeCount].pBufferInfo = &uboInfo;
    ++writeCount;

    // 1: 3D fluid temperature volume
    VkDescriptorImageInfo volInfo{};
    if (includeVolume && volumeImageView != VK_NULL_HANDLE && volumeSampler != VK_NULL_HANDLE) {
        volInfo.sampler = volumeSampler;
        volInfo.imageView = volumeImageView;
        volInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeCount].dstSet = set;
        writes[writeCount].dstBinding = 1;
        writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeCount].descriptorCount = 1;
        writes[writeCount].pImageInfo = &volInfo;
        ++writeCount;
    }

    // 2-5: Palettes
    VkDescriptorImageInfo infernoInfo{};
    infernoInfo.sampler = paletteSampler;
    infernoInfo.imageView = infernoView;
    infernoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo viridisInfo = infernoInfo;
    viridisInfo.imageView = viridisView;
    VkDescriptorImageInfo inferno2Info = infernoInfo;
    inferno2Info.imageView = inferno2View;
    VkDescriptorImageInfo parulaInfo = infernoInfo;
    parulaInfo.imageView = parulaView;
    const VkDescriptorImageInfo* paletteInfos[4] = {
        &infernoInfo, &viridisInfo, &inferno2Info, &parulaInfo};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[writeCount].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[writeCount].dstSet = set;
        writes[writeCount].dstBinding = 2 + i;
        writes[writeCount].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[writeCount].descriptorCount = 1;
        writes[writeCount].pImageInfo = paletteInfos[i];
        ++writeCount;
    }

    vkUpdateDescriptorSets(
        vulkanDevice.getDevice(), writeCount, writes.data(), 0, nullptr);
}

void HeatVolumeRenderer::updateStaticDescriptorSets() {
    if (descriptorSets.empty() || descriptorSetLayout == VK_NULL_HANDLE) {
        return;
    }
    for (size_t i = 0; i < descriptorSets.size(); ++i) {
        writeDescriptorSet(descriptorSets[i], static_cast<uint32_t>(i), false);
    }
}

void HeatVolumeRenderer::updateVolumeDescriptorSets() {
    if (descriptorSetLayout == VK_NULL_HANDLE ||
        volumeImageView == VK_NULL_HANDLE || volumeSampler == VK_NULL_HANDLE) {
        return;
    }

    const size_t frameCount = descriptorSets.empty()
        ? static_cast<size_t>(maxFramesInFlight)
        : descriptorSets.size();
    if (frameCount == 0) {
        return;
    }

    // Allocate fresh sets instead of mutating sets that may still be referenced
    // by pending command buffers.
    descriptorSets.resize(frameCount, VK_NULL_HANDLE);
    for (size_t i = 0; i < frameCount; ++i) {
        VkDescriptorSet set = allocateDescriptorSet();
        if (set == VK_NULL_HANDLE) {
            continue;
        }
        writeDescriptorSet(set, static_cast<uint32_t>(i), true);
        descriptorSets[i] = set;
    }
}

void HeatVolumeRenderer::setVolumeTexture(
    VkImageView view,
    VkSampler sampler,
    const glm::vec3& gridMin,
    const glm::ivec3& gridDim,
    float cellSize)
{
    const bool textureChanged =
        view != volumeImageView || sampler != volumeSampler;
    volumeImageView = view;
    volumeSampler = sampler;
    volumeGridMin = gridMin;
    volumeGridDim = gridDim;
    volumeCellSize = cellSize;

    if (textureChanged) {
        updateVolumeDescriptorSets();
    }
}

bool HeatVolumeRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass) {
    std::vector<char> vertShaderCode;
    std::vector<char> fragShaderCode;
    if (!readFile("shaders/heat_volume_vert.spv", vertShaderCode) ||
        !readFile("shaders/heat_volume_frag.spv", fragShaderCode)) {
        std::cerr << "[HeatVolumeRenderer] Failed to read shader files." << std::endl;
        return false;
    }

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, vertShaderCode, vertShaderModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, fragShaderCode, fragShaderModule) != VK_SUCCESS) {
        if (vertShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        if (fragShaderModule != VK_NULL_HANDLE) vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "[HeatVolumeRenderer] Failed to create shader modules." << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertStageInfo{};
    vertStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStageInfo.module = vertShaderModule;
    vertStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.module = fragShaderModule;
    fragStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertStageInfo, fragStageInfo };

    // Fullscreen triangle generated in vertex shader; empty vertex input state
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(VolumePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
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

    VkResult result = vkCreateGraphicsPipelines(vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);

    return result == VK_SUCCESS;
}

void HeatVolumeRenderer::setVolumeTexture(VkImageView view, VkSampler sampler) {
    volumeImageView = view;
    volumeSampler = sampler;

    if (initialized) {
        updateVolumeDescriptorSets();
    }
}

void HeatVolumeRenderer::setVolumeGrid(const glm::vec3& gridMin, const glm::ivec3& gridDim, float cellSize) {
    volumeGridMin = gridMin;
    volumeGridDim = gridDim;
    volumeCellSize = cellSize;

    if (!hasDomain && cellSize > 0.0f && gridDim.x > 0 && gridDim.y > 0 && gridDim.z > 0) {
        glm::vec3 boxSize = glm::vec3(gridDim) * cellSize;
        glm::mat4 domainToWorld(1.0f);
        domainToWorld[0] = glm::vec4(boxSize.x, 0.0f, 0.0f, 0.0f);
        domainToWorld[1] = glm::vec4(0.0f, boxSize.y, 0.0f, 0.0f);
        domainToWorld[2] = glm::vec4(0.0f, 0.0f, boxSize.z, 0.0f);
        domainToWorld[3] = glm::vec4(gridMin, 1.0f);
        worldToDomain = glm::inverse(domainToWorld);
    }
}

void HeatVolumeRenderer::setVolumeDomain(const std::array<glm::vec3, 8>& corners) {
    const glm::vec3 origin = corners[0];
    const glm::vec3 axisX = corners[1] - origin;
    const glm::vec3 axisY = corners[2] - origin;
    const glm::vec3 axisZ = corners[4] - origin;

    glm::mat4 domainToWorld(1.0f);
    domainToWorld[0] = glm::vec4(axisX, 0.0f);
    domainToWorld[1] = glm::vec4(axisY, 0.0f);
    domainToWorld[2] = glm::vec4(axisZ, 0.0f);
    domainToWorld[3] = glm::vec4(origin, 1.0f);

    if (std::abs(glm::determinant(domainToWorld)) > 1e-12f) {
        worldToDomain = glm::inverse(domainToWorld);
        hasDomain = true;
    } else {
        worldToDomain = glm::mat4(1.0f);
        hasDomain = false;
    }
}

void HeatVolumeRenderer::render(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!initialized ||
        !enabled ||
        pipeline == VK_NULL_HANDLE ||
        pipelineLayout == VK_NULL_HANDLE ||
        volumeImageView == VK_NULL_HANDLE ||
        volumeSampler == VK_NULL_HANDLE ||
        volumeCellSize <= 0.0f ||
        frameIndex >= descriptorSets.size()) {
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[frameIndex], 0, nullptr);

    VolumePushConstants push{};
    push.worldToDomain = worldToDomain;
    push.gridMin = volumeGridMin;
    push.cellSize = volumeCellSize;
    push.gridDim = volumeGridDim;
    push.opacity = opacity;
    push.minTemperature = minTemperature;
    push.maxTemperature = maxTemperature;
    push.palette = palette;

    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(VolumePushConstants), &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace render
