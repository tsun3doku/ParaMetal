#include "RuntimeController.hpp"

#include "RenderContext.hpp"
#include "RenderSettingsManager.hpp"
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
    WindowRuntimeState& windowRuntimeState,
    RenderSettingsManager& settingsManager,
    std::atomic<bool>& simPaused) {
    if (initialized) {
        return true;
    }

    if (!render.heatSystemComputeController() || !render.modelComputeRuntime() || !render.sceneController() || !render.nodeGraph() || !render.nodeGraphController()) {
        return false;
    }

    if (!render.initializeInputPipeline(scene, settingsManager) || !render.runtime() ||
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
    this->simPaused = &simPaused;

    runtimeInputController = std::make_unique<RuntimeInputController>(windowRuntimeState, *render.inputController());
    runtimeRenderController = std::make_unique<RuntimeRenderController>(
        *render.runtime(),
        render.sync(),
        allocator,
        settingsManager,
        render.heatSystemComputeController());

    initialized = true;
    return true;
}

void RuntimeController::shutdown() {
    runtimeRenderController.reset();
    runtimeInputController.reset();
    render = nullptr;
    scene = nullptr;
    simPaused = nullptr;
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

void RuntimeController::tick(float deltaTime, uint32_t& frameCounter) {
    if (!runtimeRenderController || !runtimeInputController || !render || !scene) {
        return;
    }

    hasFrameSlot = false;

    runtimeInputController->tick(deltaTime);

    NodeGraphController* graphController = render->nodeGraphController();
    graphController->tick();

    scene->cameraController().tick(deltaTime);
    const bool playbackPaused = simPaused && simPaused->load(std::memory_order_acquire);
    const bool allowHeatSolve = !playbackPaused && graphController->compiledState().isValid;
    const RuntimeRenderFrameResult renderResult = runtimeRenderController->renderFrame(allowHeatSolve, frameCounter);
    hasFrameSlot = renderResult.submitted;
    frameSlot = renderResult.frameSlot;
}

bool RuntimeController::hasLastFrameSlot() const {
    return hasFrameSlot;
}

uint32_t RuntimeController::lastFrameSlot() const {
    return frameSlot;
}
