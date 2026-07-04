#pragma once

#include "framegraph/FramePass.hpp"
#include "framegraph/FrameGraphTypes.hpp"

class VkFrameGraphRuntime;
class IntrinsicRenderer;

namespace render {

class HeatOverlayRenderer;
class VoronoiOverlayRenderer;

class SurfacePass : public Pass {
public:
    SurfacePass(
        VkFrameGraphRuntime& frameGraphRuntime,
        HeatOverlayRenderer& heatOverlayRenderer,
        VoronoiOverlayRenderer& voronoiOverlayRenderer,
        IntrinsicRenderer& intrinsicRenderer,
        framegraph::PassId passId,
        uint32_t maxFramesInFlight);

    const char* name() const override;
    void create() override;
    void resize(VkExtent2D extent) override;
    void updateDescriptors() override;
    void record(const FrameContext& context, const SceneView& view, const RenderFlags& flags, RenderServices& services) override;
    void destroy() override;

private:
    VkFrameGraphRuntime& frameGraphRuntime;
    HeatOverlayRenderer& heatOverlayRenderer;
    VoronoiOverlayRenderer& voronoiOverlayRenderer;
    IntrinsicRenderer& intrinsicRenderer;
    framegraph::PassId passId{};
    uint32_t maxFramesInFlight = 0;
    bool ready = false;
};

} // namespace render
