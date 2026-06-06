#include "RemeshSystem.hpp"

#include "util/GeometryUtils.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanDevice.hpp"

RemeshSystem::RemeshSystem(
    Remesher& remesher,
    VulkanDevice& vulkanDevice,
    ModelRegistry& resourceManager)
    : vulkanDevice(vulkanDevice),
      resourceManager(resourceManager),
      remesher(remesher) {
}

RemeshSystem::~RemeshSystem() {
    disable();
}

void RemeshSystem::setSourceGeometry(const std::vector<float>& updatedPointPositions, const std::vector<uint32_t>& updatedTriangleIndices) {
    pointPositions = updatedPointPositions;
    triangleIndices = updatedTriangleIndices;
}

void RemeshSystem::setParams(int nextIterations, float nextMinAngleDegrees, float nextMaxEdgeLength, float nextStepSize) {
    iterations = nextIterations;
    minAngleDegrees = nextMinAngleDegrees;
    maxEdgeLength = nextMaxEdgeLength;
    stepSize = nextStepSize;
}

void RemeshSystem::setRuntimeModelId(uint32_t nextRuntimeModelId) {
    runtimeModelId = nextRuntimeModelId;
}

bool RemeshSystem::ensureConfigured() {
    resetState(true);

    RemeshResult remeshResult{};
    if (!remesher.remesh(
            pointPositions,
            triangleIndices,
            iterations,
            minAngleDegrees,
            maxEdgeLength,
            stepSize,
            remeshResult)) {
        return false;
    }

    if (remeshResult.intrinsicGpuResources.viewS == VK_NULL_HANDLE ||
        remeshResult.intrinsicGpuResources.intrinsicTriangleBuffer == VK_NULL_HANDLE ||
        remeshResult.intrinsicGpuResources.intrinsicVertexBuffer == VK_NULL_HANDLE) {
        remesher.cleanupGpuResources(remeshResult.intrinsicGpuResources);
        return false;
    }

    geometryPositions = toVec3Array(pointPositions);
    geometryTriangleIndices = triangleIndices;
    intrinsicMesh = remeshResult.intrinsicMesh;
    intrinsicGpuResources = remeshResult.intrinsicGpuResources;
    ready = true;
    return true;
}

void RemeshSystem::disable() {
    resetState(true);
}

void RemeshSystem::resetState(bool waitForDevice) {
    if (waitForDevice && ready) {
        vkDeviceWaitIdle(vulkanDevice.getDevice());
    }

    remesher.cleanupGpuResources(intrinsicGpuResources);
    geometryPositions.clear();
    geometryTriangleIndices.clear();
    intrinsicMesh = {};
    ready = false;
}
