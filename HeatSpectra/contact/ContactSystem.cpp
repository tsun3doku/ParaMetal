#include "ContactSystem.hpp"

#include "ContactSystemRuntime.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

ContactSystem::ContactSystem(
    VulkanDevice& vulkanDeviceRef,
    MemoryAllocator& memoryAllocatorRef,
    CommandPool& renderCommandPoolRef)
    : vulkanDevice(vulkanDeviceRef),
      memoryAllocator(memoryAllocatorRef),
      renderCommandPool(renderCommandPoolRef),
      runtime(std::make_unique<ContactSystemRuntime>()) {
}

ContactSystem::~ContactSystem() {
    disable();
}

void ContactSystem::setParams(float minNormalDot, float contactRadius) {
    if (!runtime) {
        return;
    }

    runtime->setParams(minNormalDot, contactRadius);
}

void ContactSystem::setModelAState(
    const std::array<float, 16>& localToWorld,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    uint32_t runtimeModelId) {
    if (!runtime) {
        return;
    }

    runtime->setModelAState(localToWorld, intrinsicMesh, runtimeModelId);
}

void ContactSystem::setModelBState(
    const std::array<float, 16>& localToWorld,
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh,
    uint32_t runtimeModelId) {
    if (!runtime) {
        return;
    }

    runtime->setModelBState(localToWorld, intrinsicMesh, runtimeModelId);
}

void ContactSystem::setModelBTriangleIndices(const std::vector<uint32_t>& triangleIndices) {
    if (!runtime) {
        return;
    }

    runtime->setModelBTriangleIndices(triangleIndices);
}

void ContactSystem::ensureConfigured() {
    if (!runtime || !runtime->needsRebuild()) {
        return;
    }

    runtime->buildCoupling(vulkanDevice, memoryAllocator, renderCommandPool);
}

void ContactSystem::disable() {
    if (runtime) {
        runtime->clear(memoryAllocator);
    }
}

const ContactCoupling* ContactSystem::getContactCoupling() const {
    return runtime ? runtime->getContactCoupling() : nullptr;
}

VkBuffer ContactSystem::getContactPairBuffer() const {
    return runtime ? runtime->getContactPairBuffer() : VK_NULL_HANDLE;
}

VkDeviceSize ContactSystem::getContactPairBufferOffset() const {
    return runtime ? runtime->getContactPairBufferOffset() : 0;
}

const std::vector<ContactLineVertex>& ContactSystem::getOutlineVertices() const {
    static const std::vector<ContactLineVertex> empty;
    return runtime ? runtime->getOutlineVertices() : empty;
}

const std::vector<ContactLineVertex>& ContactSystem::getCorrespondenceVertices() const {
    static const std::vector<ContactLineVertex> empty;
    return runtime ? runtime->getCorrespondenceVertices() : empty;
}

bool ContactSystem::hasContact() const {
    return runtime ? runtime->hasContact() : false;
}