#include "RuntimeController.hpp"

#include "ComputeContext.hpp"
#include "NodeGraphContext.hpp"
#include "RenderContext.hpp"
#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
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
        interaction,
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
    runtimeInput.reset();
    sceneControllerState.reset();
    inputControllerState.reset();
    render = nullptr;
    compute = nullptr;
    nodeGraph = nullptr;
    scene = nullptr;
    interaction = nullptr;
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
    if (!runtimeRenderController || !runtimeInput || !render || !compute || !nodeGraph || !scene) {
        return;
    }

    runtimeInput->tick();

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
