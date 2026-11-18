#include "CFDSliceRenderer.hpp"

#include "cfd/CFDGpuStructs.hpp"
#include "util/Structs.hpp"
#include "util/file_utils.h"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"
#include "vulkan/VulkanBuffer.hpp"

#include <array>
#include <cstddef>
#include <iostream>

namespace render {

CFDSliceRenderer::CFDSliceRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uboManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uboManager),
      commandPool(commandPool) {
}

CFDSliceRenderer::~CFDSliceRenderer() {
    cleanup();
}

void CFDSliceRenderer::initialize(VkRenderPass renderPass, uint32_t subpass, uint32_t maxFramesInFlight) {
    if (initialized) {
        cleanup();
    }

    if (!createPaletteTextures() ||
        !createDescriptorSetLayout() ||
        !createDescriptorPool(maxFramesInFlight) ||
        !createDescriptorSets(maxFramesInFlight) ||
        !createPipelineLayout() ||
        !createPipeline(renderPass, subpass, false, fillPipeline) ||
        !createPipeline(renderPass, subpass, true, wireframePipeline)) {
        cleanup();
        return;
    }

    initialized = true;
}

bool CFDSliceRenderer::createPaletteTextures() {
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

void CFDSliceRenderer::destroyPaletteTextures() {
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

bool CFDSliceRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 8> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    for (uint32_t i = 1; i <= 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    for (uint32_t i = 4; i <= 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "CFDSliceRenderer: Failed to create descriptor set layout" << std::endl;
        return false;
    }

    return true;
}

bool CFDSliceRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    const uint32_t setCount = maxFramesInFlight * 16u;

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = setCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = setCount * 3;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = setCount * 4;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = setCount;

    descriptorPools.resize(maxFramesInFlight, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        if (vkCreateDescriptorPool(vulkanDevice.getDevice(), &poolInfo, nullptr, &descriptorPools[i]) != VK_SUCCESS) {
            std::cerr << "CFDSliceRenderer: Failed to create descriptor pool " << i << std::endl;
            return false;
        }
    }

    return true;
}

bool CFDSliceRenderer::createDescriptorSets(uint32_t maxFramesInFlight) {
    descriptorSets.resize(maxFramesInFlight);
    for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
        VkDescriptorSetLayout layouts[] = { descriptorSetLayout };
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPools[i];
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = layouts;

        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &set) != VK_SUCCESS) {
            std::cerr << "CFDSliceRenderer: Failed to allocate descriptor set " << i << std::endl;
            return false;
        }
        descriptorSets[i].push_back(set);
    }
    return true;
}

void CFDSliceRenderer::updateDescriptorSet(VkDescriptorSet set, uint32_t frameIndex, const RenderBinding& binding) {
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBufferManager.getUniformBuffers()[frameIndex];
    uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frameIndex];
    uboInfo.range = sizeof(UniformBufferObject);

    std::array<VkDescriptorBufferInfo, 3> bufferInfos{};
    bufferInfos[0] = { binding.velocityBuffer, binding.velocityBufferOffset, VkDeviceSize(binding.fluidNodeCount) * sizeof(glm::vec4) };
    bufferInfos[1] = { binding.centroidBuffer, binding.centroidBufferOffset, VkDeviceSize(binding.fluidNodeCount) * sizeof(glm::vec4) };
    bufferInfos[2] = { binding.gradientBuffer, binding.gradientBufferOffset, VkDeviceSize(binding.fluidNodeCount) * 3u * sizeof(glm::vec4) };

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
    const VkDescriptorImageInfo* paletteInfos[4] = {&infernoInfo, &viridisInfo, &inferno2Info, &parulaInfo};

    std::array<VkWriteDescriptorSet, 8> writes{};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].descriptorCount = 1;
    writes[0].pBufferInfo = &uboInfo;

    for (uint32_t i = 0; i < 3; ++i) {
        writes[1 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1 + i].dstSet = set;
        writes[1 + i].dstBinding = 1 + i;
        writes[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1 + i].descriptorCount = 1;
        writes[1 + i].pBufferInfo = &bufferInfos[i];
    }

    for (uint32_t i = 0; i < 4; ++i) {
        writes[4 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[4 + i].dstSet = set;
        writes[4 + i].dstBinding = 4 + i;
        writes[4 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[4 + i].descriptorCount = 1;
        writes[4 + i].pImageInfo = paletteInfos[i];
    }

    vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

bool CFDSliceRenderer::createPipelineLayout() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(cfd::SlicePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(vulkanDevice.getDevice(), &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "CFDSliceRenderer: Failed to create pipeline layout" << std::endl;
        return false;
    }
    return true;
}

bool CFDSliceRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass, bool wireframe, VkPipeline& outPipeline) {
    std::vector<char> vertShaderCode;
    std::vector<char> fragShaderCode;
    const char* vertPath = wireframe ? "shaders/cfd_slice_mesh_wire_vert.spv" : "shaders/cfd_slice_mesh_fill_vert.spv";
    const char* fragPath = wireframe ? "shaders/cfd_slice_mesh_wire_frag.spv" : "shaders/cfd_slice_mesh_fill_frag.spv";
    if (!readFile(vertPath, vertShaderCode) || !readFile(fragPath, fragShaderCode)) {
        std::cerr << "CFDSliceRenderer: Failed to read shader files" << std::endl;
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
        std::cerr << "CFDSliceRenderer: Failed to create shader modules" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStages[2]{};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertShaderModule;
    shaderStages[0].pName = "main";

    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragShaderModule;
    shaderStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkVertexInputBindingDescription vertexBinding{};
    VkVertexInputAttributeDescription attributes[2]{};
    vertexBinding.binding = 0;
    vertexBinding.stride = sizeof(cfd::CFDSliceVertex);
    vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    uint32_t attributeCount = 1;
    if (!wireframe) {
        attributes[1] = {1, 0, VK_FORMAT_R32_UINT, uint32_t(offsetof(cfd::CFDSliceVertex, fluidNode))};
        attributeCount = 2;
    }
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &vertexBinding;
    vertexInput.vertexAttributeDescriptionCount = attributeCount;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = wireframe ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
    rasterizer.depthBiasEnable = wireframe ? VK_TRUE : VK_FALSE;
    rasterizer.depthBiasConstantFactor = -2.0f;
    rasterizer.depthBiasSlopeFactor = -1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = wireframe ? VK_FALSE : VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorAttachments[1] = {};
    colorAttachments[0].colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorAttachments[0].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = colorAttachments;

    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

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
    pipelineInfo.subpass = subpass;

    const VkResult result = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outPipeline);

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "CFDSliceRenderer: Failed to create graphics pipeline" << std::endl;
        outPipeline = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool CFDSliceRenderer::isRenderable(const RenderBinding& binding) const {
    return binding.bindingKey != 0 &&
        binding.velocityBuffer != VK_NULL_HANDLE &&
        binding.gradientBuffer != VK_NULL_HANDLE &&
        binding.centroidBuffer != VK_NULL_HANDLE &&
        binding.fluidNodeCount != 0 &&
        binding.sliceGeometry != nullptr;
}

bool CFDSliceRenderer::updateGeometry(
    uint64_t key,
    uint32_t frameIndex,
    const RenderBinding& binding,
    uint32_t axis,
    double coordinate) {
    auto& frames = geometryByBinding[key];
    if (frames.size() < descriptorSets.size()) frames.resize(descriptorSets.size());
    if (frameIndex >= frames.size() || !binding.sliceGeometry) return false;
    GeometryBuffers& buffers = frames[frameIndex];
    const double scale = glm::length(glm::dvec3(binding.domainMax) - glm::dvec3(binding.domainMin));
    if (buffers.failed && buffers.sourceHash == binding.sliceGeometry->geometryHash &&
        buffers.axis == axis && buffers.coordinate == coordinate) return false;
    if (buffers.sourceHash == binding.sliceGeometry->geometryHash &&
        buffers.axis == axis && buffers.coordinate == coordinate) return true;

    cfd::CFDSliceMesh mesh;
    std::string error;
    if (!cfd::buildCFDSliceMesh(*binding.sliceGeometry, axis, coordinate, scale, mesh, &error)) {
        std::cerr << "[CFDSlice] mesh generation failed: " << error << std::endl;
        buffers.sourceHash = binding.sliceGeometry->geometryHash;
        buffers.axis = axis;
        buffers.coordinate = coordinate;
        buffers.failed = true;
        return false;
    }

    freeBuffer(memoryAllocator, buffers.triangleBuffer, buffers.triangleOffset);
    freeBuffer(memoryAllocator, buffers.lineBuffer, buffers.lineOffset);
    buffers.triangleVertexCount = buffers.lineVertexCount = 0;
    const VkDeviceSize alignment = memoryAllocator.computeBufferAlignment(
        vulkanDevice, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if ((!mesh.triangles.empty() && uploadDeviceBuffer(memoryAllocator, commandPool, mesh.triangles.data(),
            mesh.triangles.size() * sizeof(cfd::CFDSliceVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            alignment, buffers.triangleBuffer, buffers.triangleOffset) != VK_SUCCESS) ||
        (!mesh.perimeterLines.empty() && uploadDeviceBuffer(memoryAllocator, commandPool, mesh.perimeterLines.data(),
            mesh.perimeterLines.size() * sizeof(cfd::CFDSliceVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            alignment, buffers.lineBuffer, buffers.lineOffset) != VK_SUCCESS)) {
        freeBuffer(memoryAllocator, buffers.triangleBuffer, buffers.triangleOffset);
        freeBuffer(memoryAllocator, buffers.lineBuffer, buffers.lineOffset);
        return false;
    }
    buffers.triangleVertexCount = static_cast<uint32_t>(mesh.triangles.size());
    buffers.lineVertexCount = static_cast<uint32_t>(mesh.perimeterLines.size());
    buffers.sourceHash = binding.sliceGeometry->geometryHash;
    buffers.axis = axis;
    buffers.coordinate = coordinate;
    buffers.failed = false;
    return true;
}

void CFDSliceRenderer::render(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    VkExtent2D viewportExtent,
    const std::vector<RenderBinding>& slices,
    uint32_t palette,
    float minSpeed,
    float maxSpeed) {
    if (!initialized ||
        fillPipeline == VK_NULL_HANDLE ||
        wireframePipeline == VK_NULL_HANDLE ||
        pipelineLayout == VK_NULL_HANDLE ||
        frameIndex >= descriptorSets.size()) {
        return;
    }

    uint32_t bindingIndex = 0;
    for (const RenderBinding& binding : slices) {
        if (!isRenderable(binding)) {
            continue;
        }

        const uint32_t axis = binding.planeAxis;

        auto& frameSets = descriptorSets[frameIndex];
        if (bindingIndex >= frameSets.size()) {
            VkDescriptorSetLayout layout = descriptorSetLayout;
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = descriptorPools[frameIndex];
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &layout;
            VkDescriptorSet extraSet = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, &extraSet) != VK_SUCCESS) return;
            frameSets.push_back(extraSet);
        }
        const VkDescriptorSet set = frameSets[bindingIndex++];
        updateDescriptorSet(set, frameIndex, binding);

        const float planeCoord =
            binding.domainMin[axis] + binding.planeFraction * (binding.domainMax[axis] - binding.domainMin[axis]);

        if (!updateGeometry(binding.bindingKey, frameIndex, binding, axis, planeCoord)) {
            continue;
        }

        cfd::SlicePushConstants pushConstants{};
        pushConstants.viewport = glm::ivec4(static_cast<int>(viewportExtent.width), static_cast<int>(viewportExtent.height), 0, 0);
        pushConstants.counts = glm::uvec4(binding.fluidNodeCount, 0u, 0u, 0u);
        pushConstants.plane = glm::vec4(planeCoord, 0.0f, 0.0f, 0.0f);
        pushConstants.range = glm::vec4(minSpeed, maxSpeed, 0.0f, 0.0f);
        pushConstants.options = glm::uvec4(axis, palette, 0u, 0u);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fillPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0,
            1,
            &set,
            0,
            nullptr);
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(pushConstants),
            &pushConstants);
        GeometryBuffers& buffers = geometryByBinding[binding.bindingKey][frameIndex];
        if (buffers.triangleVertexCount != 0) {
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &buffers.triangleBuffer, &buffers.triangleOffset);
            vkCmdDraw(commandBuffer, buffers.triangleVertexCount, 1, 0, 0);
        }
        if (binding.showWireframe && buffers.lineVertexCount != 0) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wireframePipeline);
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &buffers.lineBuffer, &buffers.lineOffset);
            vkCmdDraw(commandBuffer, buffers.lineVertexCount, 1, 0, 0);
        }
    }
}

void CFDSliceRenderer::cleanup() {
    VkDevice device = vulkanDevice.getDevice();

    if (fillPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, fillPipeline, nullptr);
        fillPipeline = VK_NULL_HANDLE;
    }
    if (wireframePipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, wireframePipeline, nullptr);
    wireframePipeline = VK_NULL_HANDLE;
    for (auto& [key, frames] : geometryByBinding) {
        (void)key;
        for (GeometryBuffers& buffers : frames) {
            freeBuffer(memoryAllocator, buffers.triangleBuffer, buffers.triangleOffset);
            freeBuffer(memoryAllocator, buffers.lineBuffer, buffers.lineOffset);
        }
    }
    geometryByBinding.clear();
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
    descriptorSets.clear();
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    destroyPaletteTextures();

    initialized = false;
}

} // namespace render
