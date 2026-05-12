#include "HeatSystemSimRuntime.hpp"

#include "heat/HeatGpuStructs.hpp"
#include "util/Structs.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <cstring>
#include <iostream>

static void freeRuntimeBuffer(MemoryAllocator& memoryAllocator, VkBuffer& buffer, VkDeviceSize& offset) {
    if (buffer != VK_NULL_HANDLE) {
        memoryAllocator.free(buffer, offset);
        buffer = VK_NULL_HANDLE;
        offset = 0;
    }
}

bool HeatSystemSimRuntime::initialize(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator) {
    if (isInitialized()) {
        reset();
        return true;
    }

    cleanup(memoryAllocator);

    if (createUniformBuffer(memoryAllocator, vulkanDevice, sizeof(heat::TimeUniform), timeBuffer, timeBufferOffset, &mappedTimeData) != VK_SUCCESS) {
        cleanup(memoryAllocator);
        return false;
    }

    reset();
    return true;
}

void HeatSystemSimRuntime::reset() {
    if (mappedTimeData) std::memset(mappedTimeData, 0, sizeof(heat::TimeUniform));
}

void HeatSystemSimRuntime::cleanup(MemoryAllocator& memoryAllocator) {
    freeRuntimeBuffer(memoryAllocator, timeBuffer, timeBufferOffset);
    mappedTimeData = nullptr;
}

heat::TimeUniform* HeatSystemSimRuntime::getMappedTimeData() const {
    return static_cast<heat::TimeUniform*>(mappedTimeData);
}
