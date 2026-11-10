#include "RuntimeController.hpp"

#include "ComputeContext.hpp"
#include "NodeGraphContext.hpp"
#include "RenderContext.hpp"
#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "ViewportEditController.hpp"
#include "app/RawMouseInput.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "simgraph/SimGraphRuntime.hpp"
#include "render/WindowRuntimeState.hpp"
#include "render/RenderRuntime.hpp"
#include "render/SceneRenderer.hpp"
#include "render/HeatOverlayRenderer.hpp"
#include "scene/CameraController.hpp"
#include "interaction/InteractionContext.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/SceneController.hpp"
#include "vulkan/ModelRegistry.hpp"


RuntimeController::RuntimeController() = default;

RuntimeController::~RuntimeController() {
    shutdown();
}

bool RuntimeController::initialize(
    RenderContext& render,
    ComputeContext& compute,
    NodeGraphContext& nodeGraph,
    SceneContext& scene,
    InteractionContext& interaction,
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
    this->interaction = &interaction;
    this->windowRuntimeState = &windowRuntimeState;
    this->rawMouseInput = rawMouseInput;

    sceneControllerState = std::make_unique<SceneController>(
        *scene.modelRegistry(),
        scene.cameraController());
    sceneControllerState->setModelComputeController(compute.modelComputeController());
    sceneControllerState->focusOnVisibleModel();

    if (!initializeInputPipeline(windowRuntimeState)) {
        shutdown();
        return false;
    }

    viewportEdits = std::make_unique<ViewportEditController>(
        interaction,
        *nodeGraph.nodeGraphController(),
        *sceneControllerState,
        *scene.modelRegistry());
    runtimeRenderController = std::make_unique<RuntimeRenderController>(
        *render.runtime());

    initialized = true;
    return true;
}

bool RuntimeController::initializeInputPipeline(WindowRuntimeState& windowRuntimeState) {
    if (!render || !scene || !interaction || !render->runtime()) {
        return false;
    }

    RenderRuntime* renderRuntime = render->runtime();
    auto* modelRegistry = scene->modelRegistry();
    if (!modelRegistry) {
        return false;
    }

    inputControllerState = std::make_unique<InputController>(
        scene->cameraController(),
        *interaction,
        renderRuntime->getModelSelection(),
        *modelRegistry,
        windowRuntimeState);

    if (!renderRuntime->initializeFrameController(*scene, *interaction)) {
        inputControllerState.reset();
        return false;
    }

    return true;
}

void RuntimeController::shutdown() {
    runtimeRenderController.reset();
    viewportEdits.reset();
    sceneControllerState.reset();
    inputControllerState.reset();
    render = nullptr;
    compute = nullptr;
    nodeGraph = nullptr;
    scene = nullptr;
    interaction = nullptr;
    windowRuntimeState = nullptr;
    rawMouseInput = nullptr;
    rawDragActive = false;
    initialized = false;
}

bool RuntimeController::isInitialized() const {
    return initialized;
}

void RuntimeController::simulateFrame(bool allowSimulation, const SimTickContext& simContext) {
    SimGraphRuntime* simGraph = compute->simGraphRuntime();
    if (simGraph == nullptr) {
        return;
    }

    HeatSystemComputeController* heat = compute->heatSystemComputeController();

    simGraph->setEnabled(allowSimulation);
    const SimTickResult simResult = simGraph->tick(simContext);
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
    const app::RenderSettings& renderSettings,
    const SimTickContext& simContext) {
    if (!runtimeRenderController || !viewportEdits || !render || !compute || !nodeGraph || !scene) {
        return;
    }

    dispatchInputEvents();
    updateRawMouseDrag();
    viewportEdits->tick();

    if (render->runtime() && interaction) {
        const auto& markerCtrl = interaction->markerInteractionController();
        if (auto* overlay = render->runtime()->getSceneRenderer().getHeatOverlayRenderer()) {
            overlay->setSelectedProbe(markerCtrl.getSelectedProbeId(), markerCtrl.getSelectedProbeIndex());
            if (markerCtrl.hasPointerCapture()) {
                overlay->updateProbePosition(
                    markerCtrl.getSelectedProbeId(),
                    markerCtrl.getSelectedProbeIndex(),
                    markerCtrl.getLiveDragPosition());
            }
        }
    }

    NodeGraphController* graphController = nodeGraph->nodeGraphController();
    graphController->tick();

    if (render->runtime()) {
        if (auto* overlay = render->runtime()->getSceneRenderer().getHeatOverlayRenderer()) {
            if (auto* heatCtrl = compute->heatSystemComputeController()) {
                for (const auto& [nodeIdVal, node] : graphController->graphState().nodes) {
                    if (node.typeId == nodegraphtypes::HeatProbe) {
                        std::vector<float> temps;
                        if (heatCtrl->getHeatProbeTemperatures(node.id.value, temps)) {
                            overlay->updateProbeTemperatures(node.id.value, temps);
                        }
                    }
                }
            }
        }
    }

    scene->cameraController().tick(deltaTime);
    simulateFrame(
        graphController->compiledState().isValid && !graphController->runtimeCompilationPending(),
        simContext);
    runtimeRenderController->renderFrame(
        frameCounter,
        commandBuffer,
        frameIndex,
        renderSettings);
}

void RuntimeController::dispatchInputEvents() {
    if (windowRuntimeState == nullptr || !inputControllerState) {
        return;
    }

    std::vector<WindowInputEvent> inputEvents;
    windowRuntimeState->consumeInputEvents(inputEvents);
    for (const WindowInputEvent& inputEvent : inputEvents) {
        switch (inputEvent.type) {
        case WindowInputEventType::Scroll:
            inputControllerState->handleScrollInput(inputEvent.yOffset);
            break;
        case WindowInputEventType::Key:
            inputControllerState->handleKeyInput(static_cast<Qt::Key>(inputEvent.key), inputEvent.pressed, inputEvent.ctrlPressed);
            break;
        case WindowInputEventType::MousePress:
            inputControllerState->handleMouseButton(
                inputEvent.button,
                inputEvent.x,
                inputEvent.y,
                inputEvent.shiftPressed);
            break;
        case WindowInputEventType::MouseRelease:
            inputControllerState->handleMouseRelease(
                inputEvent.button,
                inputEvent.x,
                inputEvent.y);
            break;
        case WindowInputEventType::MouseMove:
            inputControllerState->handleMouseMove(
                inputEvent.x,
                inputEvent.y);
            break;
        }
    }
}

void RuntimeController::updateRawMouseDrag() {
    if (windowRuntimeState == nullptr || rawMouseInput == nullptr ||
        !rawMouseInput->isInitialized() || !inputControllerState) {
        return;
    }

    const bool middlePressed = windowRuntimeState->middleButtonPressed.load(std::memory_order_acquire);
    const bool shiftPressed = windowRuntimeState->shiftPressed.load(std::memory_order_acquire);

    if (!middlePressed) {
        rawDragActive = false;
        return;
    }
    if (!rawDragActive) {
        rawDragActive = true;
        RawMouseInput::Sample stale[64];
        while (rawMouseInput->drain(stale, 64) > 0) {
        }
        return;
    }

    float dx = 0.0f;
    float dy = 0.0f;
    RawMouseInput::Sample samples[128];
    size_t count = 0;
    do {
        count = rawMouseInput->drain(samples, 128);
        for (size_t i = 0; i < count; ++i) {
            dx += samples[i].dx;
            dy += samples[i].dy;
        }
    } while (count == 128);
    inputControllerState->processRawDrag(shiftPressed, dx, dy);
}

std::vector<ViewportCommand> RuntimeController::takePendingViewportCommands() {
    if (!inputControllerState) {
        return {};
    }
    return inputControllerState->takePendingViewportCommands();
}

bool RuntimeController::takePendingNodeParameters(
    NodeGraphNodeId& outNodeId,
    std::vector<NodeGraphParamValue>& outParameters) {
    return viewportEdits &&
        viewportEdits->takePendingNodeParameters(outNodeId, outParameters);
}
