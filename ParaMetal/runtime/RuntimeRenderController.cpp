#include "RuntimeRenderController.hpp"

#include "render/RenderConfig.hpp"
#include "render/RenderRuntime.hpp"

render::RenderFlags buildRenderFlags(const app::RenderSettings& settings) {
    render::RenderFlags flags{};
    flags.wireframeMode = static_cast<int>(settings.wireframeMode);
    flags.drawTimingOverlay = settings.gpuTimingOverlayEnabled;
    flags.drawGrid = settings.gridEnabled;
    flags.backgroundMode = static_cast<int>(settings.backgroundMode);
    flags.drawNavigationCube = settings.navigationCubeVisible;
    flags.drawAxisLabels = settings.axisLabelsVisible;
    return flags;
}

RuntimeRenderController::RuntimeRenderController(RenderRuntime& renderRuntime)
    : renderRuntime(renderRuntime) {
}

RuntimeRenderFrameResult RuntimeRenderController::renderFrame(
    uint32_t& frameCounter,
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const app::RenderSettings& settings) {
    RuntimeRenderFrameResult result{};
    result.frameSlot = frameIndex;

    const render::RenderFlags flags = buildRenderFlags(settings);

    result.submitted = renderRuntime.renderFrame(commandBuffer, frameIndex, flags);

    ++frameCounter;

    return result;
}
