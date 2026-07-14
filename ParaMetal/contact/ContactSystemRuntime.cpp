#include "ContactSystemRuntime.hpp"

#include "contact/ContactMapping.hpp"
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
    const ContactMesh& mesh,
    uint32_t runtimeModelId) {
    modelALocalToWorld = localToWorld;
    modelAMesh = mesh;
    modelARuntimeModelId = runtimeModelId;
    bindingDirty = true;
}

void ContactSystemRuntime::setModelBState(
    const std::array<float, 16>& localToWorld,
    const ContactMesh& mesh,
    uint32_t runtimeModelId) {
    modelBLocalToWorld = localToWorld;
    modelBMesh = mesh;
    modelBRuntimeModelId = runtimeModelId;
    bindingDirty = true;
}

void ContactSystemRuntime::clearPairBuffer() {
    contactPairBuffer = VK_NULL_HANDLE;
    contactPairBufferOffset = 0;
    coupling.contactPairCount = 0;
    coupling.contactPairs.clear();
}

bool ContactSystemRuntime::recreateContactPairBuffer(
    MemoryAllocator& memoryAllocator,
    VulkanDevice& vulkanDevice,
    CommandPool& commandPool,
    VkBuffer& buffer,
    VkDeviceSize& offset,
    const void* data,
    VkDeviceSize size) {
    buffer = VK_NULL_HANDLE;
    offset = 0;

    if (data == nullptr || size == 0) {
        return false;
    }

    VkDeviceSize alignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    if (uploadDeviceBuffer(memoryAllocator, commandPool, data, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, alignment, buffer, offset) != VK_SUCCESS) {
        return false;
    }

    return buffer != VK_NULL_HANDLE;
}

void ContactSystemRuntime::clearComputedState() {
    clearPairBuffer();
    coupling = {};
    couplingValid = false;
    outlineVertices.clear();
    correspondenceVertices.clear();
}

void ContactSystemRuntime::clear() {
    clearComputedState();
    minNormalDot = 0.0f;
    contactRadius = 0.0f;
    modelALocalToWorld = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    modelAMesh = {};
    modelARuntimeModelId = 0;
    modelBLocalToWorld = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    modelBMesh = {};
    modelBRuntimeModelId = 0;
    bindingDirty = false;
}

bool ContactSystemRuntime::hasValidBinding() const {
    return modelARuntimeModelId != 0 &&
        modelBRuntimeModelId != 0 &&
        modelARuntimeModelId != modelBRuntimeModelId &&
        modelAMesh.isValid() &&
        modelBMesh.isValid();
}

bool ContactSystemRuntime::computeContactPairs(std::vector<ContactPair>& outPairs) {
    outPairs.clear();
    if (modelARuntimeModelId == 0 ||
        modelBRuntimeModelId == 0 ||
        !modelAMesh.isValid() ||
        !modelBMesh.isValid()) {
        return false;
    }

    std::vector<ContactLineVertex> computedOutlineVertices;
    std::vector<ContactLineVertex> computedCorrespondenceVertices;

    buildContactPairs(
        modelAMesh,
        modelALocalToWorld,
        modelBMesh,
        modelBLocalToWorld,
        outPairs,
        computedOutlineVertices,
        computedCorrespondenceVertices,
        contactRadius,
        minNormalDot);

    outlineVertices = std::move(computedOutlineVertices);
    correspondenceVertices = std::move(computedCorrespondenceVertices);

    return hasUsableContactPairs(outPairs);
}

bool ContactSystemRuntime::buildCoupling(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool) {
    clearComputedState();

    if (!hasValidBinding()) {
        return false;
    }

    std::vector<ContactPair> pairs;
    if (!computeContactPairs(pairs) || pairs.empty()) {
        return false;
    }

    coupling.modelARuntimeModelId = modelARuntimeModelId;
    coupling.modelBRuntimeModelId = modelBRuntimeModelId;
    coupling.modelBTriangleIndices = modelBMesh.indices;
    if (coupling.modelBTriangleIndices.empty()) {
        clearComputedState();
        return false;
    }

    if (!recreateContactPairBuffer(
            memoryAllocator,
            vulkanDevice,
            commandPool,
            contactPairBuffer,
            contactPairBufferOffset,
            pairs.data(),
            sizeof(ContactPair) * pairs.size())) {
        clearComputedState();
        return false;
    }

    coupling.contactPairCount = static_cast<uint32_t>(pairs.size());
    coupling.contactPairs = std::move(pairs);
    couplingValid = coupling.isValid();
    if (couplingValid) {
        bindingDirty = false;
    }
    return couplingValid;
}