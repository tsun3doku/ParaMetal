#pragma once

#include <memory>
#include <vector>

#include "HeatReceiverRuntime.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"

class CommandPool;
class HeatSystemSimRuntime;
class MemoryAllocator;
class VulkanDevice;

class HeatSystemSurfaceRuntime {
public:
    HeatSystemSurfaceRuntime() = default;
    ~HeatSystemSurfaceRuntime();

    const std::vector<std::unique_ptr<HeatReceiverRuntime>>& getReceivers() const { return receiverRuntimes; }

    void setReceiverPayloads(
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
        const std::vector<VkDeviceSize>& receiverSurfaceGradientBufferOffsets);
    bool ensureReceiverBindings(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator);
    bool needsRebuild() const { return receiverBindingsDirty; }
    void cleanup();

private:
    std::vector<SupportingHalfedge::IntrinsicMesh> activeReceiverIntrinsicMeshes;
    std::vector<uint32_t> activeReceiverRuntimeModelIds;
    std::vector<VkBufferView> activeSupportingHalfedgeViews;
    std::vector<VkBufferView> activeSupportingAngleViews;
    std::vector<VkBufferView> activeHalfedgeViews;
    std::vector<VkBufferView> activeEdgeViews;
    std::vector<VkBufferView> activeTriangleViews;
    std::vector<VkBufferView> activeLengthViews;
    std::vector<VkBufferView> activeInputHalfedgeViews;
    std::vector<VkBufferView> activeInputEdgeViews;
    std::vector<VkBufferView> activeInputTriangleViews;
    std::vector<VkBufferView> activeInputLengthViews;
    std::vector<VkBuffer> activeReceiverSurfaceBuffers;
    std::vector<VkDeviceSize> activeReceiverSurfaceBufferOffsets;
    std::vector<VkBufferView> activeReceiverSurfaceBufferViews;
    std::vector<VkBuffer> activeReceiverSurfaceGradientBuffers;
    std::vector<VkDeviceSize> activeReceiverSurfaceGradientBufferOffsets;
    std::vector<std::unique_ptr<HeatReceiverRuntime>> receiverRuntimes;
    bool receiverBindingsDirty = true;
};
