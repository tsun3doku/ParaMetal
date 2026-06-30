#pragma once

#include <memory>

#include "app/AppTypes.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

class VulkanCoreContext {
public:
    ~VulkanCoreContext();

    bool initialize(const AppVulkanContext& vulkanContext);
    void shutdown();
    bool isInitialized() const;

    VulkanDevice& device();
    const VulkanDevice& device() const;
    MemoryAllocator* allocator();
    const MemoryAllocator* allocator() const;
    CommandPool* getRenderCommandPool();
    const CommandPool* getRenderCommandPool() const;
    CommandPool* getTransferCommandPool();
    const CommandPool* getTransferCommandPool() const;

private:
    VulkanDevice vulkanDevice;
    std::unique_ptr<MemoryAllocator> memoryAllocator;
    std::unique_ptr<CommandPool> renderCommandPool;
    std::unique_ptr<CommandPool> transferCommandPool;
    bool initialized = false;
};
