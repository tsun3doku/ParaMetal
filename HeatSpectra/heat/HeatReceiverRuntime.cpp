#include "HeatReceiverRuntime.hpp"

#include "heat/HeatGpuStructs.hpp"
#include "util/GeometryUtils.hpp"
#include "util/Structs.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <vector>

HeatReceiverRuntime::HeatReceiverRuntime(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    uint32_t runtimeModelId,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    VkBufferView supportingHalfedgeView,
    VkBufferView supportingAngleView,
    VkBufferView halfedgeView,
    VkBufferView edgeView,
    VkBufferView triangleView,
    VkBufferView lengthView,
    VkBufferView inputHalfedgeView,
    VkBufferView inputEdgeView,
    VkBufferView inputTriangleView,
    VkBufferView inputLengthView,
    VkBuffer externalSurfaceBuffer,
    VkDeviceSize externalSurfaceBufferOffset,
    VkBufferView externalSurfaceBufferView,
    VkBuffer externalGradientBuffer,
    VkDeviceSize externalGradientBufferOffset)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      runtimeModelId(runtimeModelId),
      intrinsicMesh(intrinsicMesh),
      supportingHalfedgeView(supportingHalfedgeView),
      supportingAngleView(supportingAngleView),
      halfedgeView(halfedgeView),
      edgeView(edgeView),
      triangleView(triangleView),
      lengthView(lengthView),
      inputHalfedgeView(inputHalfedgeView),
      inputEdgeView(inputEdgeView),
      inputTriangleView(inputTriangleView),
      inputLengthView(inputLengthView),
      surfaceBuffer(externalSurfaceBuffer),
      surfaceBufferOffset(externalSurfaceBufferOffset),
      surfaceBufferView(externalSurfaceBufferView),
      surfaceGradientBuffer(externalGradientBuffer),
      surfaceGradientBufferOffset(externalGradientBufferOffset) {
}

HeatReceiverRuntime::~HeatReceiverRuntime() {
}

void HeatReceiverRuntime::setSurfaceBuffer(VkBuffer buffer, VkDeviceSize offset) {
    surfaceBuffer = buffer;
    surfaceBufferOffset = offset;
}

void HeatReceiverRuntime::setGradientBuffer(VkBuffer buffer, VkDeviceSize offset) {
    surfaceGradientBuffer = buffer;
    surfaceGradientBufferOffset = offset;
}

void HeatReceiverRuntime::setGMLSSurfaceWeights(
    VkBuffer stencilBuffer,
    VkDeviceSize stencilBufferOffset,
    VkBuffer valueWeightBuffer,
    VkDeviceSize valueWeightBufferOffset,
    VkBuffer gradientWeightBuffer,
    VkDeviceSize gradientWeightBufferOffset) {
    gmlsSurfaceStencilBuffer = stencilBuffer;
    gmlsSurfaceStencilBufferOffset = stencilBufferOffset;
    gmlsSurfaceWeightBuffer = valueWeightBuffer;
    gmlsSurfaceWeightBufferOffset = valueWeightBufferOffset;
    gmlsSurfaceGradientWeightBuffer = gradientWeightBuffer;
    gmlsSurfaceGradientWeightBufferOffset = gradientWeightBufferOffset;
}

void HeatReceiverRuntime::updateDescriptors(
    VkDescriptorSetLayout surfaceLayout,
    VkDescriptorSetLayout gradientLayout,
    VkDescriptorPool surfacePool,
    VkBuffer tempBufferA,
    VkDeviceSize tempBufferAOffset,
    VkBuffer tempBufferB,
    VkDeviceSize tempBufferBOffset,
    VkBuffer timeBuffer,
    VkDeviceSize timeBufferOffset,
    uint32_t nodeCount,
    bool forceReallocate) {
    const size_t intrinsicVertexCount = getIntrinsicVertexCount();
    if (intrinsicVertexCount == 0 ||
        gmlsSurfaceStencilBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceWeightBuffer == VK_NULL_HANDLE ||
        gmlsSurfaceGradientWeightBuffer == VK_NULL_HANDLE ||
        surfaceBuffer == VK_NULL_HANDLE ||
        surfaceGradientBuffer == VK_NULL_HANDLE) {
        return;
    }

    // Allocate temperature pass descriptor sets (bindings: 0,1,10,11)
    if (forceReallocate) {
        surfaceComputeSetA = VK_NULL_HANDLE;
        surfaceComputeSetB = VK_NULL_HANDLE;
        surfaceGradientComputeSetA = VK_NULL_HANDLE;
        surfaceGradientComputeSetB = VK_NULL_HANDLE;
    }

    if (surfaceComputeSetA == VK_NULL_HANDLE || surfaceComputeSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> layouts = { surfaceLayout, surfaceLayout };

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = surfacePool;
        allocInfo.descriptorSetCount = 2;
        allocInfo.pSetLayouts = layouts.data();

        VkDescriptorSet sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &allocInfo, sets) != VK_SUCCESS) {
            std::cerr << "[HeatReceiverRuntime] Failed to allocate surface temperature descriptor sets" << std::endl;
            return;
        }

        surfaceComputeSetA = sets[0];
        surfaceComputeSetB = sets[1];
    }

    // Allocate gradient pass descriptor sets (bindings: 0,1,2,10,12)
    if (surfaceGradientComputeSetA == VK_NULL_HANDLE || surfaceGradientComputeSetB == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayout> gradLayouts = { gradientLayout, gradientLayout };

        VkDescriptorSetAllocateInfo gradAllocInfo{};
        gradAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        gradAllocInfo.descriptorPool = surfacePool;
        gradAllocInfo.descriptorSetCount = 2;
        gradAllocInfo.pSetLayouts = gradLayouts.data();

        VkDescriptorSet gradSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        if (vkAllocateDescriptorSets(vulkanDevice.getDevice(), &gradAllocInfo, gradSets) != VK_SUCCESS) {
            std::cerr << "[HeatReceiverRuntime] Failed to allocate surface gradient descriptor sets" << std::endl;
            return;
        }

        surfaceGradientComputeSetA = gradSets[0];
        surfaceGradientComputeSetB = gradSets[1];
    }

    VkDescriptorBufferInfo surfaceInfo{ surfaceBuffer, surfaceBufferOffset, sizeof(heat::SurfacePoint) * intrinsicVertexCount };
    VkDescriptorBufferInfo gradientInfo{ surfaceGradientBuffer, surfaceGradientBufferOffset, sizeof(glm::vec4) * intrinsicVertexCount };
    VkDescriptorBufferInfo timeInfo{ timeBuffer, timeBufferOffset, sizeof(heat::TimeUniform) };
    VkDescriptorBufferInfo gmlsStencilInfo{
        gmlsSurfaceStencilBuffer,
        gmlsSurfaceStencilBufferOffset,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo gmlsValueWeightInfo{
        gmlsSurfaceWeightBuffer,
        gmlsSurfaceWeightBufferOffset,
        VK_WHOLE_SIZE
    };
    VkDescriptorBufferInfo gmlsGradientWeightInfo{
        gmlsSurfaceGradientWeightBuffer,
        gmlsSurfaceGradientWeightBufferOffset,
        VK_WHOLE_SIZE
    };

    const VkDescriptorSet tempSets[2] = { surfaceComputeSetA, surfaceComputeSetB };
    const VkDescriptorSet gradSets[2] = { surfaceGradientComputeSetA, surfaceGradientComputeSetB };
    const VkBuffer tempBuffers[2] = { tempBufferA, tempBufferB };
    const VkDeviceSize tempOffsets[2] = { tempBufferAOffset, tempBufferBOffset };

    for (uint32_t pass = 0; pass < 2; ++pass) {
        // Temperature pass: bindings 0,1,10,11
        VkDescriptorBufferInfo nodeTempInfo{ tempBuffers[pass], tempOffsets[pass], sizeof(float) * nodeCount };
        std::array<VkDescriptorBufferInfo*, 4> tempInfos = {
            &nodeTempInfo,
            &surfaceInfo,
            &gmlsStencilInfo,
            &gmlsValueWeightInfo
        };
        std::array<uint32_t, 4> tempBindings = { 0u, 1u, 10u, 11u };
        std::array<VkDescriptorType, 4> tempTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };
        std::array<VkWriteDescriptorSet, 4> tempWrites{};
        for (size_t i = 0; i < tempWrites.size(); ++i) {
            tempWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            tempWrites[i].dstSet = tempSets[pass];
            tempWrites[i].dstBinding = tempBindings[i];
            tempWrites[i].dstArrayElement = 0;
            tempWrites[i].descriptorCount = 1;
            tempWrites[i].descriptorType = tempTypes[i];
            tempWrites[i].pBufferInfo = tempInfos[i];
        }
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(tempWrites.size()), tempWrites.data(), 0, nullptr);

        // Gradient pass: bindings 0,1,2,10,12
        std::array<VkDescriptorBufferInfo*, 5> gradInfos = {
            &nodeTempInfo,
            &surfaceInfo,
            &gradientInfo,
            &gmlsStencilInfo,
            &gmlsGradientWeightInfo
        };
        std::array<uint32_t, 5> gradBindings = { 0u, 1u, 2u, 10u, 12u };
        std::array<VkDescriptorType, 5> gradTypes = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        };
        std::array<VkWriteDescriptorSet, 5> gradWrites{};
        for (size_t i = 0; i < gradWrites.size(); ++i) {
            gradWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            gradWrites[i].dstSet = gradSets[pass];
            gradWrites[i].dstBinding = gradBindings[i];
            gradWrites[i].dstArrayElement = 0;
            gradWrites[i].descriptorCount = 1;
            gradWrites[i].descriptorType = gradTypes[i];
            gradWrites[i].pBufferInfo = gradInfos[i];
        }
        vkUpdateDescriptorSets(vulkanDevice.getDevice(), static_cast<uint32_t>(gradWrites.size()), gradWrites.data(), 0, nullptr);
    }
}

void HeatReceiverRuntime::cleanup() {
    // Note: surfaceBuffer, surfaceBufferView, and surfaceGradientBuffer are owned by Voronoi, do NOT free/destroy them here
    surfaceBuffer = VK_NULL_HANDLE;
    surfaceBufferOffset = 0;
    surfaceBufferView = VK_NULL_HANDLE;
    surfaceGradientBuffer = VK_NULL_HANDLE;
    surfaceGradientBufferOffset = 0;

    gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    gmlsSurfaceStencilBufferOffset = 0;
    gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    gmlsSurfaceWeightBufferOffset = 0;
    gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    gmlsSurfaceGradientWeightBufferOffset = 0;
}

VkBufferView HeatReceiverRuntime::getSupportingHalfedgeView() const {
    return supportingHalfedgeView;
}

VkBufferView HeatReceiverRuntime::getSupportingAngleView() const {
    return supportingAngleView;
}

VkBufferView HeatReceiverRuntime::getHalfedgeView() const {
    return halfedgeView;
}

VkBufferView HeatReceiverRuntime::getEdgeView() const {
    return edgeView;
}

VkBufferView HeatReceiverRuntime::getTriangleView() const {
    return triangleView;
}

VkBufferView HeatReceiverRuntime::getLengthView() const {
    return lengthView;
}

VkBufferView HeatReceiverRuntime::getInputHalfedgeView() const {
    return inputHalfedgeView;
}

VkBufferView HeatReceiverRuntime::getInputEdgeView() const {
    return inputEdgeView;
}

VkBufferView HeatReceiverRuntime::getInputTriangleView() const {
    return inputTriangleView;
}

VkBufferView HeatReceiverRuntime::getInputLengthView() const {
    return inputLengthView;
}
