#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "runtime/RuntimeProducts.hpp"

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class PointComputeRuntime {
public:
    struct Config {
        uint64_t socketKey = 0;
        std::vector<glm::vec4> positions;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        uint64_t computeHash = 0;
    };

    PointComputeRuntime(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, CommandPool& commandPool);
    ~PointComputeRuntime();

    void configure(const Config& config);
    void disable(uint64_t socketKey);
    void disableAll();

    bool exportProduct(uint64_t socketKey, PointProduct& outProduct) const;

private:
    struct SystemInstance {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        uint32_t pointCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        uint64_t computeHash = 0;
    };

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    std::unordered_map<uint64_t, SystemInstance> activeSystems;
};
