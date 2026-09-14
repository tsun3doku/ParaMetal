#include "VolumePass.hpp"

#include <iostream>

#include "framegraph/FrameGraphPasses.hpp"
#include "framegraph/VkFrameGraphRuntime.hpp"

namespace render {

VolumePass::VolumePass(
    VulkanDevice& device,
    MemoryAllocator& memoryAllocator,
    VkFrameGraphRuntime& frameGraphRuntime,
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool,
    uint32_t framesInFlight,
    framegraph::PassId passId)
    : vulkanDevice(device),
      memoryAllocator(memoryAllocator),
      frameGraphRuntime(frameGraphRuntime),
      uniformBufferManager(uniformBufferManager),
      commandPool(commandPool),
      maxFramesInFlight(framesInFlight),
      passId(passId) {
}

VolumePass::~VolumePass() = default;

const char* VolumePass::name() const {
    return framegraph::passes::Volume.data();
}

void VolumePass::create() {
    ready = false;
    destroy();

    heatVolumeRenderer = std::make_unique<HeatVolumeRenderer>(
        vulkanDevice, memoryAllocator, uniformBufferManager, commandPool);

    if (!heatVolumeRenderer || !heatVolumeRenderer->initialize(
            frameGraphRuntime.getRenderPass(),
            framegraph::toIndex(passId),
            maxFramesInFlight)) {
        std::cerr << "[VolumePass] Failed to initialize heat volume renderer." << std::endl;
        destroy();
        return;
    }

    ready = true;
}

void VolumePass::resize(VkExtent2D extent) {
    (void)extent;
}

void VolumePass::updateDescriptors() {
}

void VolumePass::record(
    const FrameContext& context,
    const SceneView& view,
    const RenderFlags& flags,
    RenderServices& services) {
    (void)view;
    (void)flags;
    (void)services;

    if (!ready || !heatVolumeRenderer) {
        return;
    }

    heatVolumeRenderer->render(context.commandBuffer, context.currentFrame);
}

void VolumePass::destroy() {
    ready = false;
    if (heatVolumeRenderer) {
        heatVolumeRenderer->cleanup();
        heatVolumeRenderer.reset();
    }
}

} // namespace render
