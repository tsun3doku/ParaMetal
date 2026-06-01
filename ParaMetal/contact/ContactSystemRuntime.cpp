#include "ContactSystemRuntime.hpp"

#include "contact/ContactSampling.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <vector>

bool ContactSystemRuntime::hasUsableContactPairs(const std::vector<ContactPair>& pairs) const {
    for (const ContactPair& pair : pairs) {
        if (pair.contactArea > 0.0f) {
            return true;
        }
    }
    return false;
}

void ContactSystemRuntime::setParams(float updatedMinNormalDot, float updatedContactRadius) {
    minNormalDot = updatedMinNormalDot;
    contactRadius = updatedContactRadius;
    bindingDirty = true;
}

void ContactSystemRuntime::setModelAState(
    const std::array<float, 16>& localToWorld,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    uint32_t runtimeModelId) {
    modelALocalToWorld = localToWorld;
    modelAIntrinsicMesh = intrinsicMesh;
    modelARuntimeModelId = runtimeModelId;
    bindingDirty = true;
}

void ContactSystemRuntime::setModelBState(
    const std::array<float, 16>& localToWorld,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    uint32_t runtimeModelId) {
    modelBLocalToWorld = localToWorld;
    modelBIntrinsicMesh = intrinsicMesh;
    modelBRuntimeModelId = runtimeModelId;
    bindingDirty = true;
}

void ContactSystemRuntime::setModelBTriangleIndices(const std::vector<uint32_t>& triangleIndices) {
    modelBTriangleIndices = triangleIndices;
    bindingDirty = true;
}

void ContactSystemRuntime::clearPairBuffer(MemoryAllocator& memoryAllocator) {
    freeBuffer(memoryAllocator, contactPairBuffer, contactPairBufferOffset);
    coupling.contactPairCount = 0;
    coupling.contactPairs.clear();
}

bool ContactSystemRuntime::recreateContactPairBuffer(
    MemoryAllocator& memoryAllocator,
    VulkanDevice& vulkanDevice,
    CommandPool& commandPool,
    VkBuffer& buffer,
    VkDeviceSize& offset,
    void** mappedData,
    const void* data,
    VkDeviceSize size) {
    freeBuffer(memoryAllocator, buffer, offset);

    if (data == nullptr || size == 0) {
        if (mappedData) {
            *mappedData = nullptr;
        }
        return false;
    }

    VkDeviceSize alignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, data, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, alignment, buffer, offset) != VK_SUCCESS) {
        if (mappedData) {
            *mappedData = nullptr;
        }
        return false;
    }

    if (mappedData) {
        *mappedData = nullptr;
    }
    return buffer != VK_NULL_HANDLE;
}

void ContactSystemRuntime::clearComputedState(MemoryAllocator& memoryAllocator) {
    clearPairBuffer(memoryAllocator);
    coupling = {};
    couplingValid = false;
    outlineVertices.clear();
    correspondenceVertices.clear();
    hasContactFlag = false;
}

void ContactSystemRuntime::clear(MemoryAllocator& memoryAllocator) {
    clearComputedState(memoryAllocator);
    minNormalDot = 0.0f;
    contactRadius = 0.0f;
    modelALocalToWorld = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    modelAIntrinsicMesh = {};
    modelARuntimeModelId = 0;
    modelBLocalToWorld = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    modelBIntrinsicMesh = {};
    modelBRuntimeModelId = 0;
    modelBTriangleIndices.clear();
    bindingDirty = false;
}

bool ContactSystemRuntime::hasValidBinding() const {
    return modelARuntimeModelId != 0 &&
        modelBRuntimeModelId != 0 &&
        modelARuntimeModelId != modelBRuntimeModelId &&
        !modelAIntrinsicMesh.vertices.empty() &&
        !modelBIntrinsicMesh.vertices.empty() &&
        !modelBTriangleIndices.empty();
}

bool ContactSystemRuntime::computeContactPairs(std::vector<ContactPair>& outPairs) {
    outPairs.clear();
    if (modelARuntimeModelId == 0 ||
        modelBRuntimeModelId == 0 ||
        modelAIntrinsicMesh.vertices.empty() ||
        modelBIntrinsicMesh.vertices.empty()) {
        return false;
    }

    std::vector<std::vector<ContactPair>> modelBContactPairs;
    std::vector<const SupportingHalfedge::IntrinsicMesh*> modelBIntrinsicMeshes;
    std::vector<std::array<float, 16>> modelBLocalToWorlds;
    modelBIntrinsicMeshes.push_back(&modelBIntrinsicMesh);
    modelBLocalToWorlds.push_back(modelBLocalToWorld);

    std::vector<ContactLineVertex> computedOutlineVertices;
    std::vector<ContactLineVertex> computedCorrespondenceVertices;

    mapSurfacePoints(
        modelAIntrinsicMesh,
        modelALocalToWorld,
        modelBIntrinsicMeshes,
        modelBLocalToWorlds,
        modelBContactPairs,
        computedOutlineVertices,
        computedCorrespondenceVertices,
        contactRadius,
        minNormalDot);

    outlineVertices = std::move(computedOutlineVertices);
    correspondenceVertices = std::move(computedCorrespondenceVertices);

    if (!modelBContactPairs.empty()) {
        outPairs = std::move(modelBContactPairs.front());
    }
    return hasUsableContactPairs(outPairs);
}

bool ContactSystemRuntime::buildCoupling(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool) {
    clearComputedState(memoryAllocator);

    if (!hasValidBinding()) {
        return false;
    }

    std::vector<ContactPair> pairs;
    if (!computeContactPairs(pairs) || pairs.empty()) {
        return false;
    }

    coupling.modelARuntimeModelId = modelARuntimeModelId;
    coupling.modelBRuntimeModelId = modelBRuntimeModelId;
    coupling.modelBTriangleIndices = modelBTriangleIndices;
    if (coupling.modelBTriangleIndices.empty()) {
        clearComputedState(memoryAllocator);
        return false;
    }

    void* mappedPairData = nullptr;
    if (!recreateContactPairBuffer(
            memoryAllocator,
            vulkanDevice,
            commandPool,
            contactPairBuffer,
            contactPairBufferOffset,
            &mappedPairData,
            pairs.data(),
            sizeof(ContactPair) * pairs.size())) {
        clearComputedState(memoryAllocator);
        return false;
    }

    coupling.contactPairCount = static_cast<uint32_t>(pairs.size());
    coupling.contactPairs = std::move(pairs);
    couplingValid = coupling.isValid();
    hasContactFlag = couplingValid;
    if (couplingValid) {
        bindingDirty = false;
    }
    return couplingValid;
}
