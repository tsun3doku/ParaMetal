#pragma once

#include "framegraph/FramePass.hpp"
#include "framegraph/FrameGraphTypes.hpp"
#include "renderers/HeatVolumeRenderer.hpp"

#include <memory>
#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;
class VkFrameGraphRuntime;
class UniformBufferManager;
class CommandPool;

namespace render {

class VolumePass : public Pass {
public:
    VolumePass(
        VulkanDevice& device,
        MemoryAllocator& memoryAllocator,
        VkFrameGraphRuntime& frameGraphRuntime,
        UniformBufferManager& uniformBufferManager,
        CommandPool& commandPool,
        uint32_t framesInFlight,
        framegraph::PassId passId);
    ~VolumePass() override;

    const char* name() const override;
    void create() override;
    void resize(VkExtent2D extent) override;
    void updateDescriptors() override;
    void record(const FrameContext& context, const SceneView& view, const RenderFlags& flags, RenderServices& services) override;
    void destroy() override;

    HeatVolumeRenderer* getHeatVolumeRenderer() const { return heatVolumeRenderer.get(); }

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    VkFrameGraphRuntime& frameGraphRuntime;
    UniformBufferManager& uniformBufferManager;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 0;
    framegraph::PassId passId{};

    std::unique_ptr<HeatVolumeRenderer> heatVolumeRenderer;
    bool ready = false;
};

} // namespace render
