#pragma once

#include "contact/ContactTypes.hpp"
#include "contact/ContactMapping.hpp"
#include <vulkan/vulkan.h>

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
        const ContactMesh& mesh,
        uint32_t runtimeModelId);
    void setModelBState(
        const std::array<float, 16>& localToWorld,
        const ContactMesh& mesh,
        uint32_t runtimeModelId);
    bool buildCoupling(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);

    void clear();
    bool hasValidBinding() const;
    const ContactCoupling* getContactCoupling() const { return couplingValid ? &coupling : nullptr; }
    VkBuffer getContactPairBuffer() const { return contactPairBuffer; }
    VkDeviceSize getContactPairBufferOffset() const { return contactPairBufferOffset; }
    const std::vector<ContactLineVertex>& getOutlineVertices() const { return outlineVertices; }
    const std::vector<ContactLineVertex>& getCorrespondenceVertices() const { return correspondenceVertices; }

private:
    void clearPairBuffer();
    bool recreateContactPairBuffer(
        MemoryAllocator& memoryAllocator,
        VulkanDevice& vulkanDevice,
        CommandPool& commandPool,
        VkBuffer& buffer,
        VkDeviceSize& offset,
        const void* data,
        VkDeviceSize size);
    void clearComputedState();
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
    ContactMesh modelAMesh;
    uint32_t modelARuntimeModelId = 0;
    std::array<float, 16> modelBLocalToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    ContactMesh modelBMesh;
    uint32_t modelBRuntimeModelId = 0;
    ContactCoupling coupling{};
    VkBuffer contactPairBuffer = VK_NULL_HANDLE; // non-owning; freed by RuntimeProducts
    VkDeviceSize contactPairBufferOffset = 0;
    bool couplingValid = false;
    bool bindingDirty = false;

    std::vector<ContactLineVertex> outlineVertices;
    std::vector<ContactLineVertex> correspondenceVertices;
};
