#include "SubselectionRenderer.hpp"

#include "mesh/MeshElementIds.hpp"
#include "scene/Model.hpp"
#include "util/Structs.hpp"
#include "util/file_utils.h"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanImage.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <unordered_set>

namespace {

struct SelectionPushConstant {
    glm::mat4 modelMatrix{1.0f};
    glm::vec4 color{1.0f, 0.55f, 0.12f, 0.55f};
    float pointRadiusWorld = 0.005f;
    float viewportHeight = 1.0f;
};

const glm::vec3 SelectionColor(1.0f, 0.55f, 0.12f);
constexpr float VertexPointRadiusFactor = 0.003f;

} // namespace

void SubselectionRenderer::buildLineVertices(
    const SubselectionDisplayController::Config& config,
    std::vector<Vertex>& outVertices) const {
    outVertices.clear();
    outVertices.reserve(config.linePositions.size());
    for (size_t i = 0; i < config.linePositions.size(); ++i) {
        const float fade = i < config.lineFade.size()
            ? config.lineFade[i] : 1.0f;
        outVertices.push_back({
            config.linePositions[i],
            SelectionColor,
            glm::vec3(0.0f),
            glm::vec2(fade, 0.0f)
        });
    }
}

void SubselectionRenderer::buildPointVertices(
    const SubselectionDisplayController::Config& config,
    std::vector<Vertex>& outVertices) const {
    for (const glm::vec3& position : config.pointPositions) {
        outVertices.push_back({
            position,
            SelectionColor,
            glm::vec3(0.0f),
            glm::vec2(1.0f, 0.0f)
        });
    }
}

SubselectionRenderer::SubselectionRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager)
    : vulkanDevice(device),
      memoryAllocator(allocator),
      uniformBufferManager(uniformBufferManager) {
}

SubselectionRenderer::~SubselectionRenderer() {
    cleanup();
}

void SubselectionRenderer::initialize(
    VkRenderPass renderPass,
    uint32_t subpass,
    uint32_t maxFramesInFlight) {
    if (initialized) {
        return;
    }
    if (!createDescriptorSetLayout() ||
        !createDescriptorPool(maxFramesInFlight) ||
        !createDescriptorSets(maxFramesInFlight) ||
        !createPipeline(renderPass, subpass)) {
        cleanup();
        return;
    }
    frameBuffers.assign(maxFramesInFlight, FrameBuffer{});
    initialized = true;
}

bool SubselectionRenderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = 1;
    info.pBindings = &binding;
    return vkCreateDescriptorSetLayout(
        vulkanDevice.getDevice(), &info, nullptr, &descriptorSetLayout) == VK_SUCCESS;
}

bool SubselectionRenderer::createDescriptorPool(uint32_t maxFramesInFlight) {
    VkDescriptorPoolSize size{};
    size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    size.descriptorCount = maxFramesInFlight;

    VkDescriptorPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = 1;
    info.pPoolSizes = &size;
    info.maxSets = maxFramesInFlight;
    return vkCreateDescriptorPool(
        vulkanDevice.getDevice(), &info, nullptr, &descriptorPool) == VK_SUCCESS;
}

bool SubselectionRenderer::createDescriptorSets(uint32_t maxFramesInFlight) {
    std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool = descriptorPool;
    alloc.descriptorSetCount = maxFramesInFlight;
    alloc.pSetLayouts = layouts.data();

    descriptorSets.resize(maxFramesInFlight);
    if (vkAllocateDescriptorSets(
            vulkanDevice.getDevice(), &alloc, descriptorSets.data()) != VK_SUCCESS) {
        descriptorSets.clear();
        return false;
    }

    for (uint32_t index = 0; index < maxFramesInFlight; ++index) {
        VkDescriptorBufferInfo buffer{};
        buffer.buffer = uniformBufferManager.getUniformBuffers()[index];
        buffer.offset = uniformBufferManager.getUniformBufferOffsets()[index];
        buffer.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSets[index];
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &buffer;
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), 1, &write, 0, nullptr);
    }
    return true;
}

bool SubselectionRenderer::createPipeline(VkRenderPass renderPass, uint32_t subpass) {
    std::vector<char> vertCode;
    std::vector<char> fragCode;
    if (!readFile("shaders/subselection_vert.spv", vertCode) ||
        !readFile("shaders/subselection_frag.spv", fragCode)) {
        std::cerr << "SubselectionRenderer: missing shaders" << std::endl;
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
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    const auto bindings = Vertex::getBindingDescriptions();
    const auto attributes = Vertex::getVertexAttributes();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
    vertexInput.pVertexBindingDescriptions = bindings.data();
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.lineWidth = 1.0f;
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable = VK_TRUE;
    raster.depthBiasConstantFactor = -1.0f;
    raster.depthBiasSlopeFactor = -1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_8_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    depth.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    const std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamic.pDynamicStates = dynamicStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.size = sizeof(SelectionPushConstant);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &descriptorSetLayout;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &pushRange;
    if (vkCreatePipelineLayout(
            vulkanDevice.getDevice(), &layout, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &assembly;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depth;
    pipelineInfo.pColorBlendState = &blend;
    pipelineInfo.pDynamicState = &dynamic;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = subpass;

    const VkResult result = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), vertModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), fragModule, nullptr);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
        return false;
    }

    std::vector<char> lineVertCode;
    std::vector<char> lineFragCode;
    if (!readFile("shaders/subselection_lines_vert.spv", lineVertCode) ||
        !readFile("shaders/subselection_frag.spv", lineFragCode)) {
        return false;
    }
    VkShaderModule lineVertModule = VK_NULL_HANDLE;
    VkShaderModule lineFragModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, lineVertCode, lineVertModule) != VK_SUCCESS ||
        createShaderModule(vulkanDevice, lineFragCode, lineFragModule) != VK_SUCCESS) {
        if (lineVertModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), lineVertModule, nullptr);
        }
        if (lineFragModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(vulkanDevice.getDevice(), lineFragModule, nullptr);
        }
        return false;
    }
    VkPipelineShaderStageCreateInfo lineStages[2]{};
    lineStages[0] = stages[0];
    lineStages[0].module = lineVertModule;
    lineStages[1] = stages[1];
    lineStages[1].module = lineFragModule;
    VkPipelineInputAssemblyStateCreateInfo lineAssembly = assembly;
    lineAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    VkGraphicsPipelineCreateInfo linePipelineInfo = pipelineInfo;
    linePipelineInfo.pStages = lineStages;
    linePipelineInfo.pInputAssemblyState = &lineAssembly;
    const VkResult lineResult = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &linePipelineInfo, nullptr, &linePipeline);
    if (lineResult != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineVertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineFragModule, nullptr);
        return false;
    }

    std::vector<char> pointFragCode;
    if (!readFile("shaders/subselection_points_frag.spv", pointFragCode)) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineVertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineFragModule, nullptr);
        return false;
    }
    VkShaderModule pointFragModule = VK_NULL_HANDLE;
    if (createShaderModule(vulkanDevice, pointFragCode, pointFragModule) != VK_SUCCESS) {
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineVertModule, nullptr);
        vkDestroyShaderModule(vulkanDevice.getDevice(), lineFragModule, nullptr);
        return false;
    }
    VkPipelineShaderStageCreateInfo pointStages[2] = {lineStages[0], stages[1]};
    pointStages[1].module = pointFragModule;
    VkPipelineInputAssemblyStateCreateInfo pointAssembly = assembly;
    pointAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    VkGraphicsPipelineCreateInfo pointPipelineInfo = pipelineInfo;
    pointPipelineInfo.pStages = pointStages;
    pointPipelineInfo.pInputAssemblyState = &pointAssembly;
    const VkResult pointResult = vkCreateGraphicsPipelines(
        vulkanDevice.getDevice(), VK_NULL_HANDLE, 1, &pointPipelineInfo, nullptr, &pointPipeline);
    vkDestroyShaderModule(vulkanDevice.getDevice(), lineVertModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), lineFragModule, nullptr);
    vkDestroyShaderModule(vulkanDevice.getDevice(), pointFragModule, nullptr);
    if (pointResult != VK_SUCCESS) {
        return false;
    }
    return true;
}

void SubselectionRenderer::rebuildSelection(
    const SubselectionDisplayController::Config& config,
    ModelRegistry& modelRegistry) {
    selectedIndices.clear();
    const Model* model = modelRegistry.getModel(config.runtimeModelId);
    if (!model || config.selectedModelIndexOffsets.empty()) {
        cachedRevision = config.revision;
        cachedModelId = config.runtimeModelId;
        return;
    }
    const std::vector<uint32_t>& indices = model->getRenderIndices();
    selectedIndices.reserve(config.selectedModelIndexOffsets.size());
    for (uint32_t offset : config.selectedModelIndexOffsets) {
        if (static_cast<size_t>(offset) + 2 >= indices.size()) {
            continue;
        }
        selectedIndices.push_back(indices[offset + 0]);
        selectedIndices.push_back(indices[offset + 1]);
        selectedIndices.push_back(indices[offset + 2]);
    }
    cachedRevision = config.revision;
    cachedModelId = config.runtimeModelId;
}

void SubselectionRenderer::uploadFrameIndices(
    uint32_t frameIndex,
    const SubselectionDisplayController::Config& config) {
    if (frameIndex >= frameBuffers.size()) {
        return;
    }
    FrameBuffer& frame = frameBuffers[frameIndex];
    if (frame.uploadedRevision == config.revision &&
        frame.uploadedModelId == config.runtimeModelId) {
        return;
    }

    const VkDeviceSize required =
        static_cast<VkDeviceSize>(selectedIndices.size()) * sizeof(uint32_t);
    if (required == 0) {
        frame.uploadedRevision = config.revision;
        frame.uploadedModelId = config.runtimeModelId;
        return;
    }

    if (required > frame.capacity || frame.buffer == VK_NULL_HANDLE) {
        if (frame.buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(frame.buffer, frame.offset);
        }
        const VkDeviceSize capacity = std::max<VkDeviceSize>(
            required,
            frame.capacity > 0 ? frame.capacity * 2 : 1024);
        const auto allocation = memoryAllocator.allocate(
            capacity,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        frame.buffer = allocation.first;
        frame.offset = allocation.second;
        frame.capacity = capacity;
    }

    if (frame.buffer == VK_NULL_HANDLE) {
        return;
    }
    void* mapped = memoryAllocator.getMappedPointer(frame.buffer, frame.offset);
    if (!mapped) {
        return;
    }
    std::memcpy(mapped, selectedIndices.data(), required);
    frame.uploadedRevision = config.revision;
    frame.uploadedModelId = config.runtimeModelId;
}

void SubselectionRenderer::uploadLineVertices(
    uint32_t frameIndex,
    const SubselectionDisplayController::Config& config) {
    if (frameIndex >= frameBuffers.size()) {
        return;
    }
    FrameBuffer& frame = frameBuffers[frameIndex];
    if (frame.uploadedLineRevision == config.revision &&
        frame.uploadedLineModelId == config.runtimeModelId) {
        return;
    }

    if (cachedLineRevision != config.revision ||
        cachedLineModelId != config.runtimeModelId) {
        std::vector<Vertex> vertices;
        buildLineVertices(config, vertices);
        cachedLineVertexCount = static_cast<uint32_t>(vertices.size());
        cachedPointFirstVertex = cachedLineVertexCount;
        buildPointVertices(config, vertices);
        cachedPointVertexCount =
            static_cast<uint32_t>(vertices.size()) - cachedPointFirstVertex;
        cachedLineVertexData.resize(vertices.size() * sizeof(Vertex));
        if (!vertices.empty()) {
            std::memcpy(cachedLineVertexData.data(), vertices.data(),
                cachedLineVertexData.size());
        }
        cachedLineRevision = config.revision;
        cachedLineModelId = config.runtimeModelId;
    }

    frame.lineVertexCount = cachedLineVertexCount;
    frame.pointFirstVertex = cachedPointFirstVertex;
    frame.pointVertexCount = cachedPointVertexCount;
    frame.uploadedLineRevision = config.revision;
    frame.uploadedLineModelId = config.runtimeModelId;
    if (cachedLineVertexData.empty()) {
        return;
    }

    const VkDeviceSize required = cachedLineVertexData.size();
    if (required > frame.lineCapacity || frame.lineBuffer == VK_NULL_HANDLE) {
        if (frame.lineBuffer != VK_NULL_HANDLE) {
            memoryAllocator.free(frame.lineBuffer, frame.lineOffset);
        }
        const VkDeviceSize capacity = std::max<VkDeviceSize>(
            required,
            frame.lineCapacity > 0 ? frame.lineCapacity * 2 : 4096);
        const auto allocation = memoryAllocator.allocate(
            capacity,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        frame.lineBuffer = allocation.first;
        frame.lineOffset = allocation.second;
        frame.lineCapacity = capacity;
    }

    if (frame.lineBuffer == VK_NULL_HANDLE) {
        frame.lineVertexCount = 0;
        frame.pointVertexCount = 0;
        return;
    }
    void* mapped = memoryAllocator.getMappedPointer(frame.lineBuffer, frame.lineOffset);
    if (!mapped) {
        frame.lineVertexCount = 0;
        frame.pointVertexCount = 0;
        return;
    }
    std::memcpy(mapped, cachedLineVertexData.data(), required);
}
void SubselectionRenderer::apply(
    uint64_t socketKey,
    const SubselectionDisplayController::Config& updatedConfig) {
    if (socketKey == 0 || updatedConfig.runtimeModelId == 0) {
        remove(activeSocketKey);
        return;
    }
    activeSocketKey = socketKey;
    config = updatedConfig;
    hasConfig = true;
}

void SubselectionRenderer::remove(uint64_t socketKey) {
    if (socketKey == 0 || socketKey != activeSocketKey) {
        return;
    }
    activeSocketKey = 0;
    config = {};
    hasConfig = false;
    cachedRevision = UINT64_MAX;
    cachedModelId = 0;
    cachedLineRevision = UINT64_MAX;
    cachedLineModelId = 0;
    selectedIndices.clear();
    cachedLineVertexData.clear();
    cachedLineVertexCount = 0;
    cachedPointFirstVertex = 0;
    cachedPointVertexCount = 0;
}

void SubselectionRenderer::render(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    VkExtent2D extent,
    ModelRegistry& modelRegistry) {
    if (!initialized || !hasConfig || frameIndex >= descriptorSets.size() ||
        (config.selectedModelIndexOffsets.empty() &&
         config.linePositions.empty() &&
         config.pointPositions.empty())) {
        return;
    }

    float pointRadiusWorld = 0.005f;
    glm::vec3 worldMin(0.0f);
    glm::vec3 worldMax(0.0f);
    if (modelRegistry.tryGetWorldBounds(
            config.runtimeModelId, worldMin, worldMax)) {
        const glm::vec3 size = worldMax - worldMin;
        const float maxExtent =
            std::max(size.x, std::max(size.y, size.z));
        if (maxExtent > 0.0f) {
            pointRadiusWorld = maxExtent * VertexPointRadiusFactor;
        }
    }

    if (config.selectedModelIndexOffsets.empty()) {
        uploadLineVertices(frameIndex, config);
        const FrameBuffer& lineFrame = frameBuffers[frameIndex];
        if (lineFrame.lineBuffer == VK_NULL_HANDLE ||
            (lineFrame.lineVertexCount == 0 && lineFrame.pointVertexCount == 0)) {
            return;
        }
        SelectionPushConstant push{};
        push.modelMatrix = config.modelMatrix;
        push.color = glm::vec4(SelectionColor, 1.0f);
        push.pointRadiusWorld = pointRadiusWorld;
        push.viewportHeight = static_cast<float>(extent.height);
        if (lineFrame.lineVertexCount > 0) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout,
                0,
                1,
                &descriptorSets[frameIndex],
                0,
                nullptr);
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(push),
                &push);
            vkCmdBindVertexBuffers(
                commandBuffer, 0, 1, &lineFrame.lineBuffer, &lineFrame.lineOffset);
            vkCmdDraw(commandBuffer, lineFrame.lineVertexCount, 1, 0, 0);
        }
        if (lineFrame.pointVertexCount > 0) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pointPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipelineLayout,
                0,
                1,
                &descriptorSets[frameIndex],
                0,
                nullptr);
            vkCmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(push),
                &push);
            vkCmdBindVertexBuffers(
                commandBuffer, 0, 1, &lineFrame.lineBuffer, &lineFrame.lineOffset);
            vkCmdDraw(commandBuffer, lineFrame.pointVertexCount, 1,
                lineFrame.pointFirstVertex, 0);
        }
        return;
    }

    if (cachedRevision != config.revision || cachedModelId != config.runtimeModelId) {
        rebuildSelection(config, modelRegistry);
    }
    if (selectedIndices.empty() || frameIndex >= frameBuffers.size()) {
        return;
    }

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize vertexBufferOffset = 0;
    VkBuffer modelIndexBuffer = VK_NULL_HANDLE;
    VkDeviceSize modelIndexBufferOffset = 0;
    uint32_t modelIndexCount = 0;
    if (!modelRegistry.tryGetRenderGeometry(
            config.runtimeModelId,
            vertexBuffer,
            vertexBufferOffset,
            modelIndexBuffer,
            modelIndexBufferOffset,
            modelIndexCount)) {
        return;
    }

    uploadFrameIndices(frameIndex, config);
    const FrameBuffer& frame = frameBuffers[frameIndex];
    if (frame.buffer == VK_NULL_HANDLE) {
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

    SelectionPushConstant push{};
    push.modelMatrix = config.modelMatrix;
    push.pointRadiusWorld = pointRadiusWorld;
    push.viewportHeight = static_cast<float>(extent.height);
    vkCmdPushConstants(
        commandBuffer,
        pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(push),
        &push);

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexBufferOffset);
    vkCmdBindIndexBuffer(commandBuffer, frame.buffer, frame.offset, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(
        commandBuffer,
        static_cast<uint32_t>(selectedIndices.size()),
        1,
        0,
        0,
        0);
}

void SubselectionRenderer::cleanup() {
    for (FrameBuffer& frame : frameBuffers) {
        if (frame.buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(frame.buffer, frame.offset);
        }
        if (frame.lineBuffer != VK_NULL_HANDLE) {
            memoryAllocator.free(frame.lineBuffer, frame.lineOffset);
        }
    }
    frameBuffers.clear();
    selectedIndices.clear();
    cachedLineVertexData.clear();
    cachedLineVertexCount = 0;
    cachedPointFirstVertex = 0;
    cachedPointVertexCount = 0;
    cachedRevision = UINT64_MAX;
    cachedModelId = 0;
    cachedLineRevision = UINT64_MAX;
    cachedLineModelId = 0;
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (linePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), linePipeline, nullptr);
        linePipeline = VK_NULL_HANDLE;
    }
    if (pointPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), pointPipeline, nullptr);
        pointPipeline = VK_NULL_HANDLE;
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), pipelineLayout, nullptr);
        pipelineLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), descriptorSetLayout, nullptr);
        descriptorSetLayout = VK_NULL_HANDLE;
    }
    descriptorSets.clear();
    initialized = false;
}
