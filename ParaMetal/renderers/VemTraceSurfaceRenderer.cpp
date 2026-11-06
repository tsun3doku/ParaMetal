#include "VemTraceSurfaceRenderer.hpp"

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanImage.hpp"
#include "util/Structs.hpp"
#include "util/file_utils.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool readEnvironmentFlag(const char* name) {
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, name) == 0 && value != nullptr) {
        const bool enabled = value[0] != '\0' && value[0] != '0';
        std::free(value);
        return enabled;
    }
    return false;
}

} // namespace

VemTraceSurfaceRenderer::VemTraceSurfaceRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uboManager,
    CommandPool& renderCommandPool)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uboManager),
      commandPool(renderCommandPool) {
}

VemTraceSurfaceRenderer::~VemTraceSurfaceRenderer() {
    cleanup();
}

void VemTraceSurfaceRenderer::initialize(
    VkRenderPass renderPass,
    uint32_t subpass,
    uint32_t maxFramesInFlight,
    const PaletteAssets& assets) {
    if (initialized) {
        cleanup();
    }

    paletteAssets = assets;
    if (paletteAssets.sampler == VK_NULL_HANDLE ||
        paletteAssets.inferno == VK_NULL_HANDLE ||
        paletteAssets.viridis == VK_NULL_HANDLE ||
        paletteAssets.inferno2 == VK_NULL_HANDLE ||
        paletteAssets.parula == VK_NULL_HANDLE ||
        paletteAssets.isocontour == VK_NULL_HANDLE ||
        paletteAssets.isocontourSampler == VK_NULL_HANDLE) {
        std::cerr << "VemTraceSurfaceRenderer: missing palette assets" << std::endl;
        return;
    }

    if (!createDescriptorPool(256) ||
        !createDescriptorSetLayout() ||
        !createPipeline(renderPass, subpass)) {
        cleanup();
        paletteAssets = {};
        return;
    }

    initialized = true;
}

bool VemTraceSurfaceRenderer::createDescriptorPool(uint32_t maxSets) {
    const uint32_t setCount = (maxSets > 0) ? maxSets : 256;

    std::array<VkDescriptorPoolSize, 3> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = setCount;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = setCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[2].descriptorCount = setCount * 6;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = setCount;

    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(
            vulkanDevice.getDevice(), &poolInfo, nullptr,
            &pool) != VK_SUCCESS) {
        std::cerr << "VemTraceSurfaceRenderer: failed to create descriptor pool"
                  << std::endl;
        return false;
    }
    descriptorPools.push_back(pool);
    return true;
}

bool VemTraceSurfaceRenderer::ensureDescriptorPool() {
    return !descriptorPools.empty() || createDescriptorPool(256);
}

bool VemTraceSurfaceRenderer::createDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 7> bindings{};

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    for (uint32_t i = 2; i <= 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(
            vulkanDevice.getDevice(), &layoutInfo, nullptr,
            &descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "VemTraceSurfaceRenderer: failed to create descriptor set layout" << std::endl;
        return false;
    }
    return true;
}

bool VemTraceSurfaceRenderer::createPipeline(
    VkRenderPass renderPass,
    uint32_t subpass) {
    std::vector<char> vertShaderCode;
    std::vector<char> fragShaderCode;
    if (!readFile("shaders/vem_trace_surface_vert.spv", vertShaderCode) ||
        !readFile("shaders/vem_trace_surface_frag.spv", fragShaderCode)) {
        std::cerr << "VemTraceSurfaceRenderer: failed to read shader files" << std::endl;
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
        std::cerr << "VemTraceSurfaceRenderer: failed to create shader modules" << std::endl;
        return false;
    }

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        vertShaderStageInfo, fragShaderStageInfo};

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(topology::VemTraceSurfaceVertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attributes{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = offsetof(topology::VemTraceSurfaceVertex, positionModel);
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = offsetof(topology::VemTraceSurfaceVertex, normalModel);
    attributes[2].location = 2;
    attributes[2].binding = 0;
    attributes[2].format = VK_FORMAT_R32_UINT;
    attributes[2].offset = offsetof(topology::VemTraceSurfaceVertex, traceIndex);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributes.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

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
    // The clipped patch vertices round-trip through canonical space, so a
    // small camera-ward bias prevents coplanar depth rejection.
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = -1.0f;
    rasterizer.depthBiasSlopeFactor = -1.0f;
    rasterizer.depthBiasClamp = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    const bool disableDepth =
        readEnvironmentFlag("HEATSPECTRA_VEM_DISABLE_DEPTH");
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = disableDepth ? VK_FALSE : VK_TRUE;
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

    VkPipelineColorBlendAttachmentState colorBlendAttachments[2] = {};
    for (int i = 0; i < 2; ++i) {
        colorBlendAttachments[i].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachments[i].blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 2;
    colorBlending.pAttachments = colorBlendAttachments;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR};

    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstant);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(
            vulkanDevice.getDevice(), &pipelineLayoutInfo, nullptr,
            &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "VemTraceSurfaceRenderer: failed to create pipeline layout" << std::endl;
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

    if (vkCreateGraphicsPipelines(
            vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo,
            nullptr, &pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
        std::cerr << "VemTraceSurfaceRenderer: failed to create graphics pipeline" << std::endl;
        return false;
    }

    vkDestroyShaderModule(vulkanDevice.getDevice(), vertShaderModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragShaderModule, nullptr);
    return true;
}

VkDescriptorSet VemTraceSurfaceRenderer::allocateDescriptorSet(
    VkDescriptorPool pool) {
    VkDescriptorSetLayout layouts[] = {descriptorSetLayout};
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = layouts;

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(
            vulkanDevice.getDevice(), &allocInfo, &set) != VK_SUCCESS) {
        static bool loggedOnce = false;
        if (!loggedOnce) {
            std::cerr << "VemTraceSurfaceRenderer: descriptor pool exhaustion" << std::endl;
            loggedOnce = true;
        }
    }
    return set;
}

VkDescriptorSet VemTraceSurfaceRenderer::findOrCreateDescriptorSet(
    uint32_t frameIndex,
    const RenderBinding& binding) {
    for (const CachedDescriptor& cached : cachedDescriptors) {
        if (cached.frameIndex == frameIndex &&
            cached.runtimeModelId == binding.runtimeModelId &&
            cached.traceBuffer == binding.traceTemperatureBuffer &&
            cached.traceCount == binding.traceCount) {
            return cached.descriptorSet;
        }
    }

    if (!ensureDescriptorPool()) {
        return VK_NULL_HANDLE;
    }

    VkDescriptorSet set = allocateDescriptorSet(descriptorPools.back());
    if (set == VK_NULL_HANDLE) {
        if (!createDescriptorPool(256)) {
            return VK_NULL_HANDLE;
        }
        set = allocateDescriptorSet(descriptorPools.back());
        if (set == VK_NULL_HANDLE) {
            return VK_NULL_HANDLE;
        }
    }

    updateDescriptorSet(set, frameIndex, binding);
    cachedDescriptors.push_back(
        {frameIndex, binding.runtimeModelId, binding.traceTemperatureBuffer,
         binding.traceCount, set});
    return set;
}

void VemTraceSurfaceRenderer::updateDescriptorSet(
    VkDescriptorSet set,
    uint32_t frameIndex,
    const RenderBinding& binding) {
    std::array<VkWriteDescriptorSet, 7> writes{};

    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBufferManager.getUniformBuffers()[frameIndex];
    uboInfo.offset = uniformBufferManager.getUniformBufferOffsets()[frameIndex];
    uboInfo.range = sizeof(UniformBufferObject);

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &uboInfo;

    VkDescriptorBufferInfo traceInfo{};
    traceInfo.buffer = binding.traceTemperatureBuffer;
    traceInfo.offset = 0;
    traceInfo.range = static_cast<VkDeviceSize>(binding.traceCount) * sizeof(float);

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &traceInfo;

    VkDescriptorImageInfo infernoInfo{};
    infernoInfo.sampler = paletteAssets.sampler;
    infernoInfo.imageView = paletteAssets.inferno;
    infernoInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo viridisInfo = infernoInfo;
    viridisInfo.imageView = paletteAssets.viridis;
    VkDescriptorImageInfo inferno2Info = infernoInfo;
    inferno2Info.imageView = paletteAssets.inferno2;
    VkDescriptorImageInfo parulaInfo = infernoInfo;
    parulaInfo.imageView = paletteAssets.parula;
    VkDescriptorImageInfo isocontourInfo{};
    isocontourInfo.sampler = paletteAssets.isocontourSampler;
    isocontourInfo.imageView = paletteAssets.isocontour;
    isocontourInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const VkDescriptorImageInfo* images[5] = {
        &infernoInfo, &viridisInfo, &inferno2Info, &parulaInfo, &isocontourInfo};
    for (uint32_t i = 0; i < 5; ++i) {
        writes[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2 + i].dstSet = set;
        writes[2 + i].dstBinding = 2 + i;
        writes[2 + i].descriptorCount = 1;
        writes[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2 + i].pImageInfo = images[i];
    }

    vkUpdateDescriptorSets(
        vulkanDevice.getDevice(), static_cast<uint32_t>(writes.size()),
        writes.data(), 0, nullptr);
}

void VemTraceSurfaceRenderer::render(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const std::vector<RenderBinding>& surfaces) {
    if (!initialized || pipeline == VK_NULL_HANDLE ||
        pipelineLayout == VK_NULL_HANDLE) {
        if (!surfaces.empty()) {
            static bool loggedUninitialized = false;
            if (!loggedUninitialized) {
                loggedUninitialized = true;
                std::cerr << "[VemTraceSurfaceRenderer] draw requested but renderer is not initialized"
                          << std::endl;
            }
        }
        return;
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    for (const RenderBinding& binding : surfaces) {
        if (binding.runtimeModelId == 0 ||
            binding.vertexBuffer == VK_NULL_HANDLE ||
            binding.traceTemperatureBuffer == VK_NULL_HANDLE ||
            binding.vertexCount == 0 ||
            binding.traceCount == 0) {
            continue;
        }

        VkDescriptorSet set = findOrCreateDescriptorSet(frameIndex, binding);
        if (set == VK_NULL_HANDLE) {
            continue;
        }

        static const bool debugPattern =
            readEnvironmentFlag("HEATSPECTRA_VEM_DEBUG_PATTERN");

        PushConstant push{};
        push.modelMatrix = binding.modelMatrix;
        push.sourceParams = glm::vec4(
            0.0f, debugPattern ? 1.0f : 0.0f, 0.0f, 0.0f);
        push.palette = palette;
        push.minTemperature = minTemperature;
        push.maxTemperature = maxTemperature;
        push.isocontourSpacing = isocontourSpacing;
        push.traceCount = binding.traceCount;

        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
            1, &set, 0, nullptr);
        vkCmdPushConstants(
            commandBuffer, pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(PushConstant), &push);
        vkCmdBindVertexBuffers(
            commandBuffer, 0, 1, &binding.vertexBuffer,
            &binding.vertexBufferOffset);
        vkCmdDraw(commandBuffer, binding.vertexCount, 1, 0, 0);
    }
}

void VemTraceSurfaceRenderer::cleanup() {
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
    cachedDescriptors.clear();
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }

    paletteAssets = {};
    initialized = false;
}
