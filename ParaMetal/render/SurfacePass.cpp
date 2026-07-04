#include "SurfacePass.hpp"

#include "HeatOverlayRenderer.hpp"
#include "VoronoiOverlayRenderer.hpp"
#include "renderers/IntrinsicRenderer.hpp"
#include "framegraph/FrameGraphPasses.hpp"
#include "framegraph/FrameGraphTypes.hpp"
#include "framegraph/VkFrameGraphRuntime.hpp"

namespace render {

SurfacePass::SurfacePass(
    VkFrameGraphRuntime& frameGraphRuntime,
    HeatOverlayRenderer& heatOverlayRenderer,
    VoronoiOverlayRenderer& voronoiOverlayRenderer,
    IntrinsicRenderer& intrinsicRenderer,
    framegraph::PassId passId,
    uint32_t maxFramesInFlight)
    : frameGraphRuntime(frameGraphRuntime),
      heatOverlayRenderer(heatOverlayRenderer),
      voronoiOverlayRenderer(voronoiOverlayRenderer),
      intrinsicRenderer(intrinsicRenderer),
      passId(passId),
      maxFramesInFlight(maxFramesInFlight) {
}

const char* SurfacePass::name() const {
    return framegraph::passes::Surface.data();
}

void SurfacePass::create() {
    heatOverlayRenderer.initializeSurface(
        frameGraphRuntime.getRenderPass(),
        framegraph::toIndex(passId),
        maxFramesInFlight);
    voronoiOverlayRenderer.initializeSurface(
        frameGraphRuntime.getRenderPass(),
        framegraph::toIndex(passId),
        maxFramesInFlight);
    intrinsicRenderer.initializeSurface(
        frameGraphRuntime.getRenderPass(),
        maxFramesInFlight,
        framegraph::toIndex(passId));
    ready = true;
}

void SurfacePass::resize(VkExtent2D extent) {
    (void)extent;
}

void SurfacePass::updateDescriptors() {
}

void SurfacePass::record(const FrameContext& context, const SceneView& view, const RenderFlags& flags, RenderServices& services) {
    (void)view;
    (void)flags;
    (void)services;
    if (!ready) {
        return;
    }

    VkCommandBuffer commandBuffer = context.commandBuffer;
    const uint32_t frameIndex = context.currentFrame;

    heatOverlayRenderer.renderSurface(commandBuffer, frameIndex);
    voronoiOverlayRenderer.renderSurface(commandBuffer, frameIndex);
    intrinsicRenderer.renderSurface(commandBuffer, frameIndex);
}

void SurfacePass::destroy() {
    ready = false;
}

} // namespace render
