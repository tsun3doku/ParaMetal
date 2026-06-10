#pragma once

#include "heat/HeatDomainRuntime.hpp"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class HeatPointsRuntime : public HeatDomainRuntime {
public:
    HeatPointsRuntime(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool,
        const std::vector<uint32_t>& boundaryConditions,
        const std::vector<float>& fixedTemperatures,
        float initialTemperature);
    ~HeatPointsRuntime();

    bool hasSurface() const override { return false; }

    uint32_t getPointCount() const { return pointCount; }
    const std::vector<uint32_t>& getBoundaryConditions() const { return boundaryConditions; }
    const std::vector<float>& getFixedTemperatures() const { return fixedTemperatures; }

    void setPositionBuffer(VkBuffer buffer, VkDeviceSize offset, uint32_t count);
    VkBuffer getPositionBuffer() const { return positionBuffer; }
    VkDeviceSize getPositionBufferOffset() const { return positionBufferOffset; }

    void cleanup();

private:
    std::vector<uint32_t> boundaryConditions;
    std::vector<float> fixedTemperatures;
    VkBuffer positionBuffer = VK_NULL_HANDLE;
    VkDeviceSize positionBufferOffset = 0;
    uint32_t pointCount = 0;
};
