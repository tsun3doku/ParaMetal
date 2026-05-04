#include "HeatSystemSurfaceRuntime.hpp"

#include "HeatReceiverRuntime.hpp"
#include "HeatSystemSimRuntime.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <algorithm>
#include <iostream>

HeatSystemSurfaceRuntime::~HeatSystemSurfaceRuntime() = default;

void HeatSystemSurfaceRuntime::setReceiverPayloads(
    const std::vector<SupportingHalfedge::IntrinsicMesh>& receiverIntrinsicMeshes,
    const std::vector<uint32_t>& receiverRuntimeModelIds,
    const std::vector<VkBufferView>& supportingHalfedgeViews,
    const std::vector<VkBufferView>& supportingAngleViews,
    const std::vector<VkBufferView>& halfedgeViews,
    const std::vector<VkBufferView>& edgeViews,
    const std::vector<VkBufferView>& triangleViews,
    const std::vector<VkBufferView>& lengthViews,
    const std::vector<VkBufferView>& inputHalfedgeViews,
    const std::vector<VkBufferView>& inputEdgeViews,
    const std::vector<VkBufferView>& inputTriangleViews,
    const std::vector<VkBufferView>& inputLengthViews,
    const std::vector<VkBuffer>& receiverSurfaceBuffers,
    const std::vector<VkDeviceSize>& receiverSurfaceBufferOffsets,
    const std::vector<VkBufferView>& receiverSurfaceBufferViews,
    const std::vector<VkBuffer>& receiverSurfaceGradientBuffers,
    const std::vector<VkDeviceSize>& receiverSurfaceGradientBufferOffsets) {
    activeReceiverIntrinsicMeshes = receiverIntrinsicMeshes;
    activeReceiverRuntimeModelIds = receiverRuntimeModelIds;
    activeSupportingHalfedgeViews = supportingHalfedgeViews;
    activeSupportingAngleViews = supportingAngleViews;
    activeHalfedgeViews = halfedgeViews;
    activeEdgeViews = edgeViews;
    activeTriangleViews = triangleViews;
    activeLengthViews = lengthViews;
    activeInputHalfedgeViews = inputHalfedgeViews;
    activeInputEdgeViews = inputEdgeViews;
    activeInputTriangleViews = inputTriangleViews;
    activeInputLengthViews = inputLengthViews;
    activeReceiverSurfaceBuffers = receiverSurfaceBuffers;
    activeReceiverSurfaceBufferOffsets = receiverSurfaceBufferOffsets;
    activeReceiverSurfaceBufferViews = receiverSurfaceBufferViews;
    activeReceiverSurfaceGradientBuffers = receiverSurfaceGradientBuffers;
    activeReceiverSurfaceGradientBufferOffsets = receiverSurfaceGradientBufferOffsets;
    receiverBindingsDirty = true;
}

bool HeatSystemSurfaceRuntime::ensureReceiverBindings(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator) {
    if (!receiverBindingsDirty) {
        return true;
    }

    cleanup();

    const size_t receiverCount = std::min({
        activeReceiverIntrinsicMeshes.size(),
        activeReceiverRuntimeModelIds.size(),
        activeSupportingHalfedgeViews.size(),
        activeSupportingAngleViews.size(),
        activeHalfedgeViews.size(),
        activeEdgeViews.size(),
        activeTriangleViews.size(),
        activeLengthViews.size(),
        activeInputHalfedgeViews.size(),
        activeInputEdgeViews.size(),
        activeInputTriangleViews.size(),
        activeInputLengthViews.size(),
        activeReceiverSurfaceBuffers.size(),
        activeReceiverSurfaceBufferViews.size(),
        activeReceiverSurfaceGradientBuffers.size()
    });
    receiverRuntimes.reserve(receiverCount);

    for (size_t index = 0; index < receiverCount; ++index) {
        const uint32_t runtimeModelId = activeReceiverRuntimeModelIds[index];
        if (runtimeModelId == 0 ||
            activeSupportingHalfedgeViews[index] == VK_NULL_HANDLE ||
            activeSupportingAngleViews[index] == VK_NULL_HANDLE ||
            activeHalfedgeViews[index] == VK_NULL_HANDLE ||
            activeEdgeViews[index] == VK_NULL_HANDLE ||
            activeTriangleViews[index] == VK_NULL_HANDLE ||
            activeLengthViews[index] == VK_NULL_HANDLE ||
            activeInputHalfedgeViews[index] == VK_NULL_HANDLE ||
            activeInputEdgeViews[index] == VK_NULL_HANDLE ||
            activeInputTriangleViews[index] == VK_NULL_HANDLE ||
            activeInputLengthViews[index] == VK_NULL_HANDLE ||
            activeReceiverSurfaceBuffers[index] == VK_NULL_HANDLE ||
            activeReceiverSurfaceBufferViews[index] == VK_NULL_HANDLE ||
            activeReceiverSurfaceGradientBuffers[index] == VK_NULL_HANDLE) {
            continue;
        }

        auto receiverRuntime = std::make_unique<HeatReceiverRuntime>(
            vulkanDevice,
            memoryAllocator,
            runtimeModelId,
            activeReceiverIntrinsicMeshes[index],
            activeSupportingHalfedgeViews[index],
            activeSupportingAngleViews[index],
            activeHalfedgeViews[index],
            activeEdgeViews[index],
            activeTriangleViews[index],
            activeLengthViews[index],
            activeInputHalfedgeViews[index],
            activeInputEdgeViews[index],
            activeInputTriangleViews[index],
            activeInputLengthViews[index],
            activeReceiverSurfaceBuffers[index],
            activeReceiverSurfaceBufferOffsets[index],
            activeReceiverSurfaceBufferViews[index],
            activeReceiverSurfaceGradientBuffers[index],
            activeReceiverSurfaceGradientBufferOffsets[index]);
        receiverRuntimes.push_back(std::move(receiverRuntime));
    }

    receiverBindingsDirty = false;
    return true;
}

void HeatSystemSurfaceRuntime::cleanup() {
    for (auto& receiverRuntime : receiverRuntimes) {
        if (receiverRuntime) {
            receiverRuntime->cleanup();
        }
    }
    receiverRuntimes.clear();
    receiverBindingsDirty = true;
}
