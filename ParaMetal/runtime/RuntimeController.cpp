#include "RuntimeController.hpp"

#include "ComputeContext.hpp"
#include "NodeGraphContext.hpp"
#include "RenderContext.hpp"
#include "RuntimeRenderController.hpp"
#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "ViewportEditController.hpp"
#include "app/RawMouseInput.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "simgraph/SimGraphRuntime.hpp"
#include "render/WindowRuntimeState.hpp"
#include "render/RenderRuntime.hpp"
#include "render/SceneRenderer.hpp"
#include "render/HeatOverlayRenderer.hpp"
#include "render/CFDOverlayRenderer.hpp"
#include "scene/CameraController.hpp"
#include "scene/DomainFaceInteractionController.hpp"
#include "scene/InputController.hpp"
#include "interaction/InteractionContext.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/SceneController.hpp"
#include "scene/SubselectionInteractionController.hpp"
#include "runtime/RenderContext.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <algorithm>
#include <array>
#include <utility>

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

    sceneControllerState = std::make_unique<SceneController>();
    nodeGraph.setSceneController(sceneControllerState.get());

    if (!initializeInputPipeline(windowRuntimeState)) {
        shutdown();
        return false;
    }

    viewportEdits = std::make_unique<ViewportEditController>(
        interaction,
        *nodeGraph.nodeGraphController(),
        *sceneControllerState,
        *scene.modelRegistry(),
        render.runtime()->getModelSelection(),
        scene.cameraController(),
        windowRuntimeState);
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
    interaction->setSubselectionPickService(&renderRuntime->getSubselectionPickService());

    if (!renderRuntime->initializeFrameController(*scene, *interaction)) {
        inputControllerState.reset();
        return false;
    }

    return true;
}

void RuntimeController::shutdown() {
    runtimeRenderController.reset();
    viewportEdits.reset();
    if (nodeGraph) {
        nodeGraph->setSceneController(nullptr);
    }
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

    render::HeatOverlayRenderer* heatOverlay = nullptr;
    render::CFDOverlayRenderer* cfdOverlay = nullptr;
    DomainFaceInteractionController* faceEditing = nullptr;
    bool domainBoxValid = false;

    if (render->runtime() && interaction) {
        SceneRenderer& sceneRenderer = render->runtime()->getSceneRenderer();
        heatOverlay = sceneRenderer.getHeatOverlayRenderer();
        cfdOverlay = sceneRenderer.getCFDOverlayRenderer();
        faceEditing = &interaction->domainFaceInteractionController();

        std::array<glm::vec3, 8> domainBox{};
        bool boxSet = false;
        bool boxAmbiguous = false;
        const std::vector<HeatBoxRenderer::RenderBinding>* boxSources[2] = {
            heatOverlay ? &heatOverlay->getBoxBindings() : nullptr,
            cfdOverlay ? &cfdOverlay->getBoxBindings() : nullptr
        };
        for (const auto* boxes : boxSources) {
            if (!boxes) {
                continue;
            }
            for (const auto& binding : *boxes) {
                if (!boxSet) {
                    domainBox = binding.corners;
                    boxSet = true;
                } else if (binding.corners != domainBox) {
                    boxAmbiguous = true;
                }
            }
        }

        domainBoxValid = boxSet && !boxAmbiguous;
        faceEditing->setDomainBox(domainBox, domainBoxValid);
    }

    updateRawMouseDrag();
    scene->cameraController().tick(deltaTime);
    viewportEdits->refreshPickCamera();
    dispatchInputEvents();
    viewportEdits->tick();

    if (render->runtime() && interaction) {
        const SubselectionInteractionController& subselection =
            interaction->subselectionInteractionController();
        render->runtime()->getSceneRenderer().setSubselectionPickState(
            subselection.pickState());
        syncSubselectionDisplay(subselection);
    }

    if (faceEditing) {
        const int selectedFace = faceEditing->selectedFace();
        const int hoverFace = faceEditing->hoverFace();
        const bool highlightValid = domainBoxValid && faceEditing->editingNodeId() > 0;
        if (heatOverlay) {
            heatOverlay->setBoundaryFaceHighlight(
                selectedFace, hoverFace, faceEditing->domainCorners(), highlightValid);
        }
        if (cfdOverlay) {
            cfdOverlay->setBoundaryFaceHighlight(
                selectedFace, hoverFace, faceEditing->domainCorners(), highlightValid);
        }
    }

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
    if (sceneControllerState && scene && scene->modelRegistry()) {
        scene->modelRegistry()->setSceneExtent(
            sceneControllerState->sceneWorldExtent());
    }
    std::vector<uint32_t> displayedModelIds =
        graphController->getPackageController().displayedSourceModelIds();
    if (render->runtime()) {
        render->runtime()->getSceneRenderer().setDisplayedModelIds(
            displayedModelIds);
    }
    viewportEdits->setDisplayedSourceModelIds(std::move(displayedModelIds));
    viewportEdits->reconcileSelection();
    collectRuntimeEvents();

    if (render->runtime()) {
        if (auto* overlay = render->runtime()->getSceneRenderer().getHeatOverlayRenderer()) {
            if (const auto* heatCtrl = compute->heatSystemComputeController()) {
                heatCtrl->publishProbeTelemetry(*overlay);
            }
        }
    }

    simulateFrame(
        graphController->compiledState().isValid && !graphController->runtimeCompilationPending(),
        simContext);
    runtimeRenderController->renderFrame(
        frameCounter,
        commandBuffer,
        frameIndex,
        renderSettings);
}

void RuntimeController::syncSubselectionDisplay(
    const SubselectionInteractionController& subselection) {
    SubselectionDisplayController* subselectionDisplay =
        render ? render->subselectionDisplayController() : nullptr;
    RemeshDisplayController* remeshDisplay =
        render ? render->remeshDisplayController() : nullptr;

    if (!subselection.isActive()) {
        if (subselectionDisplay) {
            subselectionDisplay->finalizeSync();
        }
        if (remeshDisplay && appliedIntrinsicSelectionSocketKey != 0) {
            remeshDisplay->removeSelection(
                appliedIntrinsicSelectionSocketKey);
            appliedIntrinsicSelectionSocketKey = 0;
        }
        return;
    }

    const SubselectionEditState& state = subselection.editState();
    const MeshSelection& selection = subselection.displaySelection();
    const uint64_t revision = subselection.displayRevision();

    if (state.binding == SurfaceBinding::Intrinsic) {
        if (subselectionDisplay) {
            subselectionDisplay->finalizeSync();
        }
        if (remeshDisplay) {
            if (appliedIntrinsicSelectionSocketKey != 0 &&
                appliedIntrinsicSelectionSocketKey !=
                    state.displaySocketKey) {
                remeshDisplay->removeSelection(
                    appliedIntrinsicSelectionSocketKey);
            }
            remeshDisplay->applySelection(
                state.displaySocketKey, state, selection, revision);
            appliedIntrinsicSelectionSocketKey = state.displaySocketKey;
        }
    } else {
        if (remeshDisplay && appliedIntrinsicSelectionSocketKey != 0) {
            remeshDisplay->removeSelection(
                appliedIntrinsicSelectionSocketKey);
            appliedIntrinsicSelectionSocketKey = 0;
        }
        if (subselectionDisplay) {
            subselectionDisplay->applySelection(
                state.displaySocketKey, state, selection, revision);
            subselectionDisplay->finalizeSync();
        }
    }
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
            if (inputEvent.pressed &&
                inputEvent.key == Qt::Key_Escape &&
                viewportEdits) {
                viewportEdits->cancelAction();
            } else if (inputEvent.pressed &&
                (inputEvent.key == Qt::Key_Return ||
                 inputEvent.key == Qt::Key_Enter) &&
                viewportEdits) {
                viewportEdits->completeAction();
            }
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

void RuntimeController::setSelectedNodes(
    const std::vector<NodeGraphNodeId>& nodeIds,
    int primaryNodeId) {
    if (viewportEdits) {
        viewportEdits->setSelectedNodes(nodeIds, primaryNodeId);
    }
}

void RuntimeController::invalidateSelection() {
    lastInspectedNodeId = 0;
    lastInspectionGraphRevision = UINT64_MAX;
    lastInspectionEvaluationRevision = UINT64_MAX;
    lastInspectionModelRevision = UINT64_MAX;
    if (viewportEdits) {
        viewportEdits->invalidateSelection();
    }
}

void RuntimeController::reconcileSelection() {
    if (viewportEdits) {
        viewportEdits->reconcileSelection();
    }
}

bool RuntimeController::beginNodeAction(
    NodeGraphNodeId nodeId,
    uint64_t nodeInstanceId,
    uint32_t actionId) {
    return viewportEdits &&
        viewportEdits->beginAction(nodeId, nodeInstanceId, actionId);
}

bool RuntimeController::beginViewportSelection() {
    return viewportEdits && viewportEdits->beginViewportSelection();
}

bool RuntimeController::selectionCommitRequired() const {
    return viewportEdits && viewportEdits->selectionCommitRequired();
}

void RuntimeController::completeNodeAction() {
    if (viewportEdits) {
        viewportEdits->completeAction();
    }
}

void RuntimeController::cancelNodeAction() {
    if (viewportEdits) {
        viewportEdits->cancelAction();
    }
}

void RuntimeController::clearNodeActionSelection() {
    if (viewportEdits) {
        viewportEdits->clearActionSelection();
    }
}

void RuntimeController::setSubselectionElementType(int elementType) {
    if (viewportEdits) {
        viewportEdits->setSubselectionElementType(elementType);
    }
}

std::vector<ViewportCommand> RuntimeController::takePendingViewportCommands() {
    if (!inputControllerState) {
        return {};
    }
    return inputControllerState->takePendingViewportCommands();
}

bool RuntimeController::takePendingNodeParameters(
    NodeGraphNodeId& outNodeId,
    uint64_t& outOriginInstanceId,
    std::vector<NodeGraphParamValue>& outParameters) {
    return viewportEdits &&
        viewportEdits->takePendingNodeParameters(
            outNodeId, outOriginInstanceId, outParameters);
}

std::vector<RuntimeEvent> RuntimeController::takePendingRuntimeEvents() {
    std::vector<RuntimeEvent> events;
    events.swap(pendingRuntimeEvents);
    return events;
}

void RuntimeController::collectRuntimeEvents() {
    if (!viewportEdits || !nodeGraph) {
        return;
    }

    NodeActionCompleted outcome;
    while (viewportEdits->takePendingActionOutcome(outcome)) {
        pendingRuntimeEvents.emplace_back(std::move(outcome));
    }

    RuntimeNodeSelectionRequested selectionRequest;
    while (viewportEdits->takeSelectionRequest(selectionRequest)) {
        pendingRuntimeEvents.push_back(selectionRequest);
    }

    NodeGraphController* graphController = nodeGraph->nodeGraphController();
    if (!graphController) {
        return;
    }

    const int selectedNodeId = viewportEdits->selectedNode();
    const uint64_t graphRevision = graphController->graphState().revision;
    const uint64_t evaluationRevision = graphController->evaluationRevision();
    const uint64_t modelRevision = graphController->runtimeModelRevision();
    if (selectedNodeId == lastInspectedNodeId &&
        graphRevision == lastInspectionGraphRevision &&
        evaluationRevision == lastInspectionEvaluationRevision &&
        modelRevision == lastInspectionModelRevision) {
        return;
    }

    lastInspectedNodeId = selectedNodeId;
    lastInspectionGraphRevision = graphRevision;
    lastInspectionEvaluationRevision = evaluationRevision;
    lastInspectionModelRevision = modelRevision;
    if (selectedNodeId <= 0) {
        return;
    }

    NodeInspection inspection;
    if (graphController->inspect(
            NodeGraphNodeId{static_cast<uint32_t>(selectedNodeId)},
            inspection)) {
        pendingRuntimeEvents.push_back(NodeInspectionChanged{
            graphController->graphState().instanceId,
            NodeGraphNodeId{static_cast<uint32_t>(selectedNodeId)},
            std::move(inspection),
        });
    }
}
