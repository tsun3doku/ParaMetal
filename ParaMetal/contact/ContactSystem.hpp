#pragma once

#include "contact/ContactTypes.hpp"
#include "contact/ContactMapping.hpp"
#include "vulkan/CommandBufferManager.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class ContactSystemRuntime;
class MemoryAllocator;
class VulkanDevice;

class ContactSystem {
public:
    ContactSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);
    ~ContactSystem();

    void setParams(float minNormalDot, float contactRadius);
    void setModelAState(
        const std::array<float, 16>& localToWorld,
        const ContactMesh& mesh,
        uint32_t runtimeModelId);
    void setModelBState(
        const std::array<float, 16>& localToWorld,
        const ContactMesh& mesh,
        uint32_t runtimeModelId);
    void ensureConfigured();
    void disable();

    const ContactCoupling* getContactCoupling() const;
    VkBuffer getContactPairBuffer() const;
    VkDeviceSize getContactPairBufferOffset() const;
    const std::vector<ContactLineVertex>& getOutlineVertices() const;
    const std::vector<ContactLineVertex>& getCorrespondenceVertices() const;

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    std::unique_ptr<ContactSystemRuntime> runtime;
};
