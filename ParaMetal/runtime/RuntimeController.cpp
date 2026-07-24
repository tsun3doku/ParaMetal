#include "RuntimeController.hpp"

#include "RenderContext.hpp"
#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "render/WindowRuntimeState.hpp"
#include "scene/CameraController.hpp"

RuntimeController::~RuntimeController() {
    shutdown();
}

bool RuntimeController::initialize(
    RenderContext& render,
    SceneContext& scene,
    VulkanCoreContext& core,
    WindowRuntimeState& windowRuntimeState) {
    if (initialized) {
        return true;
    }

    if (!render.heatSystemComputeController() || !render.modelComputeRuntime() || !render.sceneController() || !render.nodeGraphController()) {
        return false;
    }

    if (!render.initializeInputPipeline(scene) || !render.runtime() ||
        !render.inputController()) {
        shutdown();
        return false;
    }

    auto* allocator = core.allocator();
    if (!allocator) {
        shutdown();
        return false;
    }

    this->render = &render;
    this->scene = &scene;

    runtimeInputController = std::make_unique<RuntimeInputController>(windowRuntimeState, *render.inputController());
    runtimeRenderController = std::make_unique<RuntimeRenderController>(
        *render.runtime(),
        render.sync(),
        allocator,
        render.heatSystemComputeController());

    initialized = true;
    return true;
}

void RuntimeController::shutdown() {
    runtimeRenderController.reset();
    runtimeInputController.reset();
    render = nullptr;
    scene = nullptr;
    initialized = false;
}

bool RuntimeController::isInitialized() const {
    return initialized;
}

RuntimeInputController& RuntimeController::inputController() {
    return *runtimeInputController;
}

NodeGraphController* RuntimeController::nodeGraphController() {
    return render ? render->nodeGraphController() : nullptr;
}

CameraController* RuntimeController::cameraController() {
    return scene ? &scene->cameraController() : nullptr;
}

void RuntimeController::tick(
    float deltaTime,
    uint32_t& frameCounter,
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const app::RenderSettings& renderSettings) {
    if (!runtimeRenderController || !runtimeInputController || !render || !scene) {
        return;
    }

    hasFrameSlot = false;

    runtimeInputController->tick(deltaTime);

    NodeGraphController* graphController = render->nodeGraphController();
    graphController->tick();

    scene->cameraController().tick(deltaTime);
    const bool allowHeatSolve = graphController->compiledState().isValid;
    const RuntimeRenderFrameResult renderResult = runtimeRenderController->renderFrame(
        allowHeatSolve,
        frameCounter,
        commandBuffer,
        frameIndex,
        renderSettings);
    hasFrameSlot = renderResult.submitted;
    frameSlot = renderResult.frameSlot;
}

bool RuntimeController::hasLastFrameSlot() const {
    return hasFrameSlot;
}

uint32_t RuntimeController::lastFrameSlot() const {
    return frameSlot;
}
