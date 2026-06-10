#include "HeatPointsRuntime.hpp"

#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "heat/HeatGpuStructs.hpp"

#include <array>
#include <cstring>
#include <vector>

HeatPointsRuntime::HeatPointsRuntime(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool,
    const std::vector<uint32_t>& boundaryConditions,
    const std::vector<float>& fixedTemperatures,
    float initialTemperature)
    : HeatDomainRuntime(vulkanDevice, memoryAllocator, renderCommandPool),
      boundaryConditions(boundaryConditions),
      fixedTemperatures(fixedTemperatures) {
    (void)initialTemperature;
}

HeatPointsRuntime::~HeatPointsRuntime() {
    cleanup();
}

void HeatPointsRuntime::setPositionBuffer(VkBuffer buffer, VkDeviceSize offset, uint32_t count) {
    positionBuffer = buffer;
    positionBufferOffset = offset;
    pointCount = count;
}

void HeatPointsRuntime::cleanup() {
    positionBuffer = VK_NULL_HANDLE;
    positionBufferOffset = 0;
    pointCount = 0;
    cleanupSimulationBuffers();
}
