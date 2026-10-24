#include "RuntimeController.hpp"

#include "ComputeContext.hpp"
#include "NodeGraphContext.hpp"
#include "RenderContext.hpp"
#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "simgraph/SimGraphRuntime.hpp"
#include "render/WindowRuntimeState.hpp"
#include "render/RenderRuntime.hpp"
#include "scene/CameraController.hpp"
#include "scene/InputController.hpp"
#include "scene/SceneController.hpp"
#include "scene/MaterialSystem.hpp"
#include "scene/LightingSystem.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"

RuntimeController::RuntimeController() = default;

RuntimeController::~RuntimeController() {
    shutdown();
}

bool RuntimeController::initialize(
    RenderContext& render,
    ComputeContext& compute,
    NodeGraphContext& nodeGraph,
    SceneContext& scene,
    VulkanCoreContext& core,
    WindowRuntimeState& windowRuntimeState,
    RawMouseInput* rawMouseInput) {
    if (initialized) {
        return true;
    }

    auto* allocator = core.allocator();
    if (!allocator) {
        shutdown();
        return false;
    }

    this->render = &render;
    this->compute = &compute;
    this->nodeGraph = &nodeGraph;
    this->scene = &scene;

    sceneControllerState = std::make_unique<SceneController>(
        *scene.modelRegistry(),
        scene.cameraController());
    sceneControllerState->setModelComputeController(compute.modelComputeController());
    sceneControllerState->focusOnVisibleModel();

    if (!initializeInputPipeline(windowRuntimeState)) {
        shutdown();
        return false;
    }

    runtimeInput = std::make_unique<RuntimeInput>(
        windowRuntimeState,
        *inputControllerState,
        *nodeGraph.nodeGraphController(),
        *sceneControllerState,
        *scene.modelRegistry(),
        rawMouseInput);
    runtimeRenderController = std::make_unique<RuntimeRenderController>(
        *render.runtime());

    initialized = true;
    return true;
}

bool RuntimeController::initializeInputPipeline(WindowRuntimeState& windowRuntimeState) {
    if (!render || !scene || !render->runtime()) {
        return false;
    }

    RenderRuntime* renderRuntime = render->runtime();
    auto* modelRegistry = scene->modelRegistry();
    auto* uniformBufferManager = scene->uniformBufferManager();
    auto* lightingSystem = scene->lightingSystem();
    auto* materialSystem = scene->materialSystem();
    if (!modelRegistry || !uniformBufferManager || !lightingSystem || !materialSystem) {
        return false;
    }

    inputControllerState = std::make_unique<InputController>(
        scene->cameraController(),
        renderRuntime->getGizmoController(),
        renderRuntime->getNavigationGizmoController(),
        renderRuntime->getModelSelection(),
        *modelRegistry,
        windowRuntimeState);

    FrameControllerServices frameControllerServices{
        *modelRegistry,
        *uniformBufferManager,
        renderRuntime->getModelSelection(),
        renderRuntime->getGizmoController(),
        renderRuntime->getNavigationGizmoController(),
        renderRuntime->getWireframeRenderer(),
        *inputControllerState,
        *lightingSystem,
        *materialSystem};
    if (!renderRuntime->initializeFrameController(frameControllerServices)) {
        inputControllerState.reset();
        return false;
    }

    return true;
}

void RuntimeController::shutdown() {
    runtimeRenderController.reset();
    runtimeInput.reset();
    sceneControllerState.reset();
    inputControllerState.reset();
    render = nullptr;
    compute = nullptr;
    nodeGraph = nullptr;
    scene = nullptr;
    initialized = false;
}

bool RuntimeController::isInitialized() const {
    return initialized;
}

NodeGraphController* RuntimeController::nodeGraphController() {
    return nodeGraph ? nodeGraph->nodeGraphController() : nullptr;
}

CameraController* RuntimeController::cameraController() {
    return scene ? &scene->cameraController() : nullptr;
}

void RuntimeController::simulateFrame(bool allowSimulation, uint32_t frameIndex) {
    if (render == nullptr || compute == nullptr) {
        return;
    }

    SimGraphRuntime* simGraph = compute->simGraphRuntime();
    if (simGraph == nullptr) {
        return;
    }

    HeatSystemComputeController* heat = compute->heatSystemComputeController();
    if (heat != nullptr) {
        heat->updateSerialInputs();
    }

    simGraph->setEnabled(allowSimulation);
    const SimTickResult simResult = simGraph->tick(frameIndex);
    render->runtime()->getFrameStats().setComputeGpuMs(simResult.computeGpuMs);
    render->runtime()->getFrameStats().setSimStepTotal(simResult.completedStepCount);

    if (simResult.status == SimStatus::Failed && heat != nullptr) {
        // Unblock queued consumer waits on failed sim before quiescing CUDA work
        heat->releaseWaits();
        heat->waitIdle();
    }
}

void RuntimeController::tick(
    float deltaTime,
    uint32_t& frameCounter,
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    const app::RenderSettings& renderSettings) {
    if (!runtimeRenderController || !runtimeInput || !render || !compute || !nodeGraph || !scene) {
        return;
    }

    hasFrameSlot = false;

    runtimeInput->tick();

    NodeGraphController* graphController = nodeGraph->nodeGraphController();
    graphController->tick();

    scene->cameraController().tick(deltaTime);
    simulateFrame(
        graphController->compiledState().isValid && !graphController->runtimeCompilationPending(),
        frameIndex);
    const RuntimeRenderFrameResult renderResult = runtimeRenderController->renderFrame(
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

std::vector<ViewportCommand> RuntimeController::takePendingViewportCommands() {
    if (!runtimeInput) {
        return {};
    }
    return runtimeInput->takePendingViewportCommands();
}

bool RuntimeController::takePendingNodeParameters(
    NodeGraphNodeId& outNodeId,
    std::vector<NodeGraphParamValue>& outParameters) {
    return runtimeInput &&
        runtimeInput->takePendingNodeParameters(outNodeId, outParameters);
}

SceneController* RuntimeController::sceneController() {
    return sceneControllerState.get();
}

const SceneController* RuntimeController::sceneController() const {
    return sceneControllerState.get();
}
