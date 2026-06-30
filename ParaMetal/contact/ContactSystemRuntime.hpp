#pragma once

#include "contact/ContactTypes.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"

#include <array>
#include <vector>

class MemoryAllocator;
class VulkanDevice;
class CommandPool;

class ContactSystemRuntime {
public:
    bool needsRebuild() const { return bindingDirty; }

    void setParams(float minNormalDot, float contactRadius);
    void setModelAState(
        const std::array<float, 16>& localToWorld,
        const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
        uint32_t runtimeModelId);
    void setModelBState(
        const std::array<float, 16>& localToWorld,
        const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
        uint32_t runtimeModelId);
    void setModelBTriangleIndices(const std::vector<uint32_t>& triangleIndices);
    bool buildCoupling(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);

    void clear(MemoryAllocator& memoryAllocator);
    bool hasValidBinding() const;
    const ContactCoupling* getContactCoupling() const { return couplingValid ? &coupling : nullptr; }
    VkBuffer getContactPairBuffer() const { return contactPairBuffer; }
    VkDeviceSize getContactPairBufferOffset() const { return contactPairBufferOffset; }
    const std::vector<ContactLineVertex>& getOutlineVertices() const { return outlineVertices; }
    const std::vector<ContactLineVertex>& getCorrespondenceVertices() const { return correspondenceVertices; }
    bool hasContact() const { return hasContactFlag; }

private:
    void clearPairBuffer(MemoryAllocator& memoryAllocator);
    bool recreateContactPairBuffer(
        MemoryAllocator& memoryAllocator,
        VulkanDevice& vulkanDevice,
        CommandPool& commandPool,
        VkBuffer& buffer,
        VkDeviceSize& offset,
        void** mappedData,
        const void* data,
        VkDeviceSize size);
    void clearComputedState(MemoryAllocator& memoryAllocator);
    bool computeContactPairs(std::vector<ContactPair>& outPairs);
    bool hasUsableContactPairs(const std::vector<ContactPair>& pairs) const;

    float minNormalDot = -0.65f;
    float contactRadius = 0.01f;
    std::array<float, 16> modelALocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    SupportingHalfedge::IntrinsicMesh modelAIntrinsicMesh;
    uint32_t modelARuntimeModelId = 0;
    std::array<float, 16> modelBLocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    SupportingHalfedge::IntrinsicMesh modelBIntrinsicMesh;
    uint32_t modelBRuntimeModelId = 0;
    std::vector<uint32_t> modelBTriangleIndices;
    ContactCoupling coupling{};
    VkBuffer contactPairBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactPairBufferOffset = 0;
    bool couplingValid = false;
    bool bindingDirty = false;

    std::vector<ContactLineVertex> outlineVertices;
    std::vector<ContactLineVertex> correspondenceVertices;
    bool hasContactFlag = false;
};
