#include "RuntimeSystems.hpp"
#include "telemetry/RuntimeTelemetry.hpp"

#include "app/RuntimeNotifier.hpp"
#include "app/ViewportMailbox.hpp"

#include <QtCore/QString>
#include <QtCore/QVariantList>
#include <QtCore/QVariantMap>

#include "cuda/CudaVulkanDevice.hpp"
#include "cfd/CFDSystemComputeController.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "render/RenderConfig.hpp"
#include "render/WindowRuntimeState.hpp"
#include "render/SceneRenderer.hpp"
#include "render/CFDOverlayRenderer.hpp"
#include "render/HeatOverlayRenderer.hpp"
#include "scene/DomainFaceInteractionController.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/NavigationGizmoController.hpp"
#include "scene/SubselectionInteractionController.hpp"
#include "simgraph/SimGraphRuntime.hpp"
#include "vulkan/ModelRegistry.hpp"


#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

static constexpr uint64_t RetireReclaimWaitNs = 5'000'000'000ull;

RuntimeSystems::RuntimeSystems() = default;

RuntimeSystems::~RuntimeSystems() {
    shutdown();
}

bool RuntimeSystems::initialize(WindowRuntimeState& runtimeState, const AppVulkanContext& vulkanContext) {
    if (initialized) {
        return true;
    }

    renderSettingsState = {};
    windowRuntimeState = &runtimeState;
    isShuttingDown.store(false, std::memory_order_release);
    runtimeBusy.store(false, std::memory_order_release);
    frameCounter = 0;
    timelineRuntime.reset();
    hadSimulationTimeline = false;

    if (!core.initialize(vulkanContext)) {
        cleanup();
        return false;
    }
    if (!scene.initialize(core)) {
        cleanup();
        return false;
    }
    if (!interaction.initialize(scene.cameraController(), *scene.modelRegistry(), runtimeState)) {
        cleanup();
        return false;
    }
    if (!frameSync.initialize(core.device().getDevice(), renderconfig::MaxFramesInFlight)) {
        cleanup();
        return false;
    }
    if (!rawMouseInput.initialize()) {
        std::cerr << "[RuntimeSystems] raw mouse input unavailable" << std::endl;
    }
    if (!render.initialize(core, scene, frameSync, runtimeState, vulkanContext, runtimeBusy, isShuttingDown)) {
        cleanup();
        return false;
    }
    if (!compute.initialize(core, scene, frameSync, runtimeBusy)) {
        cleanup();
        return false;
    }
    if (!nodeGraph.initialize(core, compute, render.displayControllers())) {
        cleanup();
        return false;
    }

    if (!runtimeController.initialize(
            render, compute, nodeGraph, scene, interaction, core, runtimeState, &rawMouseInput)) {
        cleanup();
        return false;
    }

    initialized = true;
    return true;
}

void RuntimeSystems::tickFrame(float deltaTime, VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!initialized || !windowRuntimeState || !runtimeController.isInitialized()) {
        return;
    }
    if (isShuttingDown.load(std::memory_order_acquire) || runtimeBusy.load(std::memory_order_acquire)) {
        return;
    }


    SimGraphRuntime* simGraph = compute.simGraphRuntime();
    const bool hasSimulationTimeline =
        simGraph != nullptr && simGraph->hasSimulationTimelineMetadata();
    if (hasSimulationTimeline) {
        const uint32_t timelineFrames = simGraph->getSimulationTimelineFrameCount();
        const float duration = simGraph->getSimulationDuration();
        if (timelineFrames > 0) {
            timelineRuntime.setFrameCount(timelineFrames + 1u);
        }
        if (duration > 0.0f && timelineFrames > 0) {
            timelineRuntime.setFps(static_cast<float>(timelineFrames) / duration);
        }

        uint32_t maxAdvanceFrame = timelineRuntime.getMaxFrame();
        if (timelineRuntime.isPlaying()) {
            const uint32_t availableFrame = simGraph->getPlaybackAvailableFrame();
            if (availableFrame != UINT32_MAX) {
                maxAdvanceFrame = std::min(maxAdvanceFrame, availableFrame);
            }
        }
        timelineRuntime.tick(deltaTime, maxAdvanceFrame);
    } else if (hadSimulationTimeline) {
        timelineRuntime.clearSimulationMetadata();
    }
    hadSimulationTimeline = hasSimulationTimeline;

    compute.tickDomainServices();

    SimTickContext context{};
    context.renderFrameSlot = frameIndex;
    context.deltaSeconds = deltaTime;
    context.playing = timelineRuntime.isPlaying();
    context.currentFrame = timelineRuntime.getCurrentFrame();
    context.resetEpoch = timelineRuntime.getResetEpoch();

    reclaimRetiredGenerations();

    runtimeController.tick(
        deltaTime,
        frameCounter,
        commandBuffer,
        frameIndex,
        renderSettingsState,
        context);
    dispatchViewportCommands();
}

void RuntimeSystems::reclaimRetiredGenerations() {
    NodeGraphController* controller = getNodeGraphController();
    if (controller == nullptr || !controller->hasRetiringGenerations()) {
        return;
    }
    if (!frameSync.waitForAllFrameFences(RetireReclaimWaitNs)) {
        std::cerr << "[RuntimeSystems] compute submissions still in flight; retiring generations retained"
                  << std::endl;
        return;
    }
    if (!cudaVulkan::waitForGpuIdle(
            core.device().getDevice(), core.device().getPhysicalDevice())) {
        std::cerr << "[RuntimeSystems] GPU idle wait failed; retiring generations retained"
                  << std::endl;
        return;
    }
    controller->clearRetiringGenerations();
}

bool RuntimeSystems::updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent) {
    if (!initialized) {
        return false;
    }
    return render.updateViewportTarget(image, format, extent);
}

bool RuntimeSystems::replaceGraphState(const NodeGraphState& state) {
    NodeGraphController* controller = getNodeGraphController();

    if (controller) {
        controller->detachRuntimeBindings();
    }

    // Device may not exist yet during early initialization
    if (!cudaVulkan::waitForGpuIdle(
            core.device().getDevice(), core.device().getPhysicalDevice())) {
        std::cerr << "[RuntimeSystems] graph replacement aborted after GPU idle wait failure"
                  << std::endl;
        return false;
    }

    if (controller) {
        controller->quiescedClear();
    }
    runtimeController.invalidateSelection();
    compute.quiescedClear();
    runtimeExport.clear();

    if (ModelRegistry* registry = scene.modelRegistry()) {
        registry->clearModels();
    }
    if (MemoryAllocator* allocator = core.allocator()) {
        allocator->trim();
    }
    if (controller) {
        controller->resetGraph(state);
    }
    return true;
}

void RuntimeSystems::applyInteractionMailbox(ViewportMailbox& mailbox, bool force) {
    std::vector<NodeGraphNodeId> selectedNodes;
    int primarySelectedNode = 0;
    if (mailbox.takeSelection(
            selectedNodes, primarySelectedNode, force)) {
        runtimeController.setSelectedNodes(
            selectedNodes, primarySelectedNode);
    }

    runtimeController.reconcileSelection();

    std::vector<NodeActionMessage> nodeActions;
    mailbox.takeNodeActions(nodeActions);
    for (const NodeActionMessage& message : nodeActions) {
        if (message.type != NodeActionMessage::Type::Begin) {
            continue;
        }
        runtimeController.beginNodeAction(
            message.nodeId,
            message.nodeInstanceId,
            message.actionId);
    }

    bool viewportMeshSelection = false;
    if (mailbox.takeViewportMeshSelection(viewportMeshSelection, force)) {
        if (viewportMeshSelection) {
            runtimeController.beginViewportSelection();
        } else {
            runtimeController.cancelNodeAction();
        }
    }

    int elementType = static_cast<int>(MeshElementType::Face);
    if (mailbox.takeSubselectionElementType(elementType, force)) {
        runtimeController.setSubselectionElementType(elementType);
    }

    for (const NodeActionMessage& message : nodeActions) {
        switch (message.type) {
        case NodeActionMessage::Type::Begin:
            break;
        case NodeActionMessage::Type::Complete:
            runtimeController.completeNodeAction();
            break;
        case NodeActionMessage::Type::Cancel:
            runtimeController.cancelNodeAction();
            break;
        case NodeActionMessage::Type::Clear:
            runtimeController.clearNodeActionSelection();
            break;
        }
    }
}

bool RuntimeSystems::applyGraphMailbox(ViewportMailbox& mailbox, bool force) {
    if (const NodeGraphState* graphState = mailbox.graphReplacement(force)) {
        if (!replaceGraphState(*graphState)) {
            return false;
        }
        mailbox.graphReplacementApplied();
        return true;
    }

    bool deltasApplied = true;
    bool graphReplaced = false;
    NodeGraphController* controller = getNodeGraphController();
    for (const NodeGraphDelta& delta : mailbox.graphDeltas()) {
        if (!controller || !controller->applyGraphDelta(delta)) {
            mailbox.clearNodeActions();
            const NodeGraphState* cachedState = mailbox.graphReplacement(true);
            Q_ASSERT(cachedState);
            if (!cachedState || !replaceGraphState(*cachedState)) {
                deltasApplied = false;
            } else {
                graphReplaced = true;
            }
            break;
        }
    }
    if (deltasApplied) {
        mailbox.graphDeltasApplied();
    }
    return graphReplaced;
}

void RuntimeSystems::dispatchViewportCommands() {
    for (ViewportCommand command : runtimeController.takePendingViewportCommands()) {
        switch (command) {
        case ViewportCommand::ToggleWireframe:
            renderSettingsState.wireframeMode = static_cast<app::WireframeMode>(
                (static_cast<int>(renderSettingsState.wireframeMode) + 1) % 3);
            break;
        case ViewportCommand::ToggleTimingOverlay:
            renderSettingsState.gpuTimingOverlayEnabled =
                !renderSettingsState.gpuTimingOverlayEnabled;
            break;
        case ViewportCommand::ToggleGrid:
            renderSettingsState.gridEnabled = !renderSettingsState.gridEnabled;
            break;
        }
    }
}

void RuntimeSystems::requestExport(int nodeId) {
    runtimeExport.request(nodeId);
}

void RuntimeSystems::requestExportProducts(int nodeId, uint64_t requestId, uint64_t graphRevision) {
    runtimeExport.requestExportProducts(nodeId, requestId, graphRevision);
}

bool RuntimeSystems::takePendingExportResult(
    int& outNodeId,
    bool& outSuccess,
    std::string& outMessage) {
    return runtimeExport.takeResult(outNodeId, outSuccess, outMessage);
}

bool RuntimeSystems::takeExportProductsReady(ExportProductsReply& outReply) {
    return runtimeExport.takeProductsReady(outReply);
}

void RuntimeSystems::processDeferredWork() {
    runtimeExport.processPending(
        getNodeGraphController(),
        compute.heatSystemComputeController(),
        compute.cfdSystemComputeController());
}

void RuntimeSystems::publishRuntimeEvents(RuntimeNotifier& notifier) {
    NodeGraphNodeId nodeId{};
    uint64_t originInstanceId = 0;
    std::vector<NodeGraphParamValue> parameters;
    while (runtimeController.takePendingNodeParameters(
            nodeId, originInstanceId, parameters)) {
        notifier.publishNodeParametersRequested(
            nodeId, originInstanceId, parameters);
    }

    for (const RuntimeEvent& event : runtimeController.takePendingRuntimeEvents()) {
        if (const auto* inspection = std::get_if<NodeInspectionChanged>(&event)) {
            QVariantList fields;
            fields.reserve(static_cast<qsizetype>(inspection->inspection.fields.size()));
            for (const NodeInspectionField& field : inspection->inspection.fields) {
                QVariantMap row;
                row.insert(QStringLiteral("name"), QString::fromStdString(field.name));
                row.insert(QStringLiteral("value"), QString::fromStdString(field.value));
                fields.push_back(row);
            }

            QVariantMap choices;
            for (const NodeParameterChoices& parameter : inspection->inspection.parameterChoices) {
                QVariantList parameterChoices;
                parameterChoices.reserve(static_cast<qsizetype>(parameter.choices.size()));
                for (const NodeChoice& choice : parameter.choices) {
                    QVariantMap row;
                    row.insert(QStringLiteral("text"), QString::fromStdString(choice.label));
                    row.insert(QStringLiteral("value"), QString::fromStdString(choice.value));
                    parameterChoices.push_back(row);
                }
                choices.insert(
                    QString::number(parameter.parameterId),
                    parameterChoices);
            }

            notifier.publishNodeInspection(
                inspection->graphInstanceId,
                static_cast<int>(inspection->nodeId.value),
                fields,
                choices);
            continue;
        }

        if (const auto* request = std::get_if<RuntimeNodeSelectionRequested>(&event)) {
            notifier.publishViewportSelectionRequested(
                request->graphInstanceId,
                request->nodeId,
                request->nodeInstanceId,
                request->additive);
            continue;
        }

        if (const auto* completed = std::get_if<NodeActionCompleted>(&event)) {
            notifier.publishNodeActionCompleted(
                static_cast<int>(completed->nodeId.value),
                completed->success,
                QString::fromStdString(completed->error));
        }
    }

    if (const NodeGraphController* graphController = getNodeGraphController()) {
        notifier.publishHeatSolveError(
            QString::fromStdString(graphController->lastPackageErrors()));
    }

    if (interaction.isInitialized()) {
        if (auto selection = interaction.domainFaceInteractionController().takePendingSelection()) {
            notifier.publishBoundaryFaceSelection(
                static_cast<int>(selection->nodeId), selection->face);
        }
        if (auto selectedNode = interaction.markerInteractionController().takePendingSelectedNodeId()) {
            notifier.notifyGraphSelectionChanged(NodeGraphNodeId{*selectedNode});
        }
        const SubselectionInteractionController& subselection =
            interaction.subselectionInteractionController();
        const bool subselectionActive = subselection.isActive();
        const int subselectionCount = subselection.isActive()
            ? static_cast<int>(subselection.selectedElements().size()) : 0;
        const int subselectionElementType = subselection.isActive()
            ? static_cast<int>(subselection.authoredSelection().elementType)
            : static_cast<int>(MeshElementType::Face);
        const bool selectionCommitRequired =
            runtimeController.selectionCommitRequired();
        if (subselectionActive != publishedSubselectionActive ||
            selectionCommitRequired != publishedSubselectionCommitRequired ||
            subselectionCount != publishedSubselectionCount ||
            subselectionElementType != publishedSubselectionElementType) {
            publishedSubselectionActive = subselectionActive;
            publishedSubselectionCommitRequired = selectionCommitRequired;
            publishedSubselectionCount = subselectionCount;
            publishedSubselectionElementType = subselectionElementType;
            notifier.publishSubselectionState(
                subselectionActive, subselectionCount, subselectionElementType,
                selectionCommitRequired);
        }
    }
}

void RuntimeSystems::shutdown() {
    if (!initialized && !core.isInitialized() && !scene.isInitialized() && !render.isInitialized() &&
        !runtimeController.isInitialized() && !windowRuntimeState) {
        return;
    }

    isShuttingDown.store(true, std::memory_order_release);
    cleanup();
    runtimeBusy.store(false, std::memory_order_release);
    initialized = false;
}

bool RuntimeSystems::isInitialized() const {
    return initialized;
}

void RuntimeSystems::noteFramePresented() {
    render.runtime()->getFrameStats().notePresentedFrame();
}

void RuntimeSystems::publishProbeTelemetry(TemperatureTelemetryReceiver& receiver) const {
    const HeatSystemComputeController* controller =
        compute.heatSystemComputeController();
    if (controller) {
        controller->publishProbeTelemetry(receiver);
    }
}

void RuntimeSystems::publishSerialTelemetry(ConnectionTelemetryReceiver& receiver) const {
    const HeatSystemComputeController* controller =
        compute.heatSystemComputeController();
    if (controller) {
        controller->publishSerialTelemetry(receiver);
    }
}

void RuntimeSystems::startProbePicking(uint32_t targetNodeId, int probeIndex) {
    if (interaction.isInitialized()) {
        interaction.startProbePicking(targetNodeId, probeIndex);
    }
}

void RuntimeSystems::cancelProbePicking() {
    if (interaction.isInitialized()) {
        interaction.cancelProbePicking();
    }
}

void RuntimeSystems::setBoundaryFaceEditing(int nodeId, bool armed) {
    if (interaction.isInitialized()) {
        interaction.setBoundaryFaceEditing(nodeId, armed);
    }
}

TimelineController* RuntimeSystems::timelineController() {
    return &timelineControllerInstance;
}

void RuntimeSystems::setWireframeMode(app::WireframeMode mode) {
    renderSettingsState.wireframeMode = mode;
}

void RuntimeSystems::setGridEnabled(bool enabled) {
    renderSettingsState.gridEnabled = enabled;
}

void RuntimeSystems::setBackgroundMode(app::BackgroundMode mode) {
    renderSettingsState.backgroundMode = mode;
}

void RuntimeSystems::setNavigationCubeVisible(bool visible) {
    renderSettingsState.navigationCubeVisible = visible;
    if (interaction.isInitialized()) {
        interaction.navigationGizmoController().setEnabled(visible);
    }
}

void RuntimeSystems::setAxisLabelsVisible(bool visible) {
    renderSettingsState.axisLabelsVisible = visible;
}

void RuntimeSystems::focusCameraOnWorldOrigin() {
    scene.cameraController().focusWorldOrigin();
}

void RuntimeSystems::setCameraProjectionMode(CameraProjectionMode mode) {
    scene.cameraController().setProjectionMode(mode);
}

void RuntimeSystems::setCameraFov(float degrees) {
    scene.cameraController().setBaseFov(degrees);
}

void RuntimeSystems::setCameraZoomSpeed(float speed) {
    scene.cameraController().setZoomSpeed(speed);
}

void RuntimeSystems::setCameraPanSpeed(float speed) {
    scene.cameraController().setPanSpeed(speed);
}

void RuntimeSystems::setCameraCull(float meters, bool linesOnly) {
    if (!std::isfinite(meters)) return;
    const float clamped = std::clamp(meters, Camera::MinNearClip, Camera::MaxNearClip);
    renderSettingsState.nearClipMeters = clamped;
    renderSettingsState.cullLinesOnly = linesOnly;
    scene.cameraController().getCamera().setNearClip(linesOnly ? Camera::DefaultNearClip : clamped);
    applyStreamlineCull();
}

void RuntimeSystems::applyStreamlineCull() {
    if (!render.runtime()) return;
    const float worldDistance = renderSettingsState.cullLinesOnly
        ? renderSettingsState.nearClipMeters * units::canonicalToWorldScale(worldUnitState)
        : 0.0f;
    render.runtime()->getSceneRenderer().setStreamlineCull(worldDistance);
}

void RuntimeSystems::setWorldUnit(int unit) {
    if (unit < static_cast<int>(units::LengthUnit::Millimeter) ||
        unit > static_cast<int>(units::LengthUnit::Meter)) return;
    const units::LengthUnit next = static_cast<units::LengthUnit>(unit);
    if (next == worldUnitState) return;
    worldUnitState = next;
    if (NodeGraphController* controller = getNodeGraphController()) {
        controller->setWorldUnit(next);
    }
    scene.cameraController().setWorldUnit(next);
    if (render.runtime()) {
        render.runtime()->getSceneRenderer().setWorldUnit(next);
    }
    applyStreamlineCull();
}

void RuntimeSystems::setHeatPaletteRange(float minimum, float maximum) {
    if (render.runtime()) {
        render.runtime()->getSceneRenderer().setHeatPaletteRange(minimum, maximum);
    }
}

void RuntimeSystems::setHeatPalette(int palette) {
    if (render.runtime()) {
        render.runtime()->getSceneRenderer().setHeatPalette(palette);
    }
}

bool RuntimeSystems::isHeatPaletteVisible() const {
    const RenderRuntime* runtimeState = render.runtime();
    return runtimeState && runtimeState->getSceneRenderer().isHeatPaletteVisible();
}

void RuntimeSystems::setVelocityPaletteRange(float minimum, float maximum) {
    if (render.runtime()) {
        render.runtime()->getSceneRenderer().setVelocityPaletteRange(minimum, maximum);
    }
}

void RuntimeSystems::setVelocityPalette(int palette) {
    if (render.runtime()) {
        render.runtime()->getSceneRenderer().setVelocityPalette(palette);
    }
}

bool RuntimeSystems::isVelocityPaletteVisible() const {
    const RenderRuntime* runtimeState = render.runtime();
    return runtimeState && runtimeState->getSceneRenderer().isVelocityPaletteVisible();
}

const app::RenderSettings& RuntimeSystems::renderSettings() const {
    return renderSettingsState;
}

CameraController* RuntimeSystems::getCameraController() {
    return &scene.cameraController();
}

ModelSelection* RuntimeSystems::getModelSelection() {
    RenderRuntime* renderRuntime = render.runtime();
    if (!renderRuntime) {
        return nullptr;
    }
    return &renderRuntime->getModelSelection();
}

NodeGraphController* RuntimeSystems::getNodeGraphController() {
    return nodeGraph.nodeGraphController();
}

void RuntimeSystems::cleanup() {
    rawMouseInput.shutdown();
    runtimeController.shutdown();

    HeatSystemComputeController* heatSystemController = compute.heatSystemComputeController();
    cfd::CFDSystemComputeController* cfdSystemController = compute.cfdSystemComputeController();
    if (heatSystemController) {
        heatSystemController->releaseWaits();
    }
    if (cfdSystemController) {
        cfdSystemController->releaseWaits();
    }

    if (!cudaVulkan::waitForGpuIdle(
            core.device().getDevice(), core.device().getPhysicalDevice())) {
        std::cerr << "[RuntimeSystems] terminal shutdown aborted after GPU idle wait failure"
                  << std::endl;
        std::abort();
    }

    nodeGraph.shutdown();
    compute.shutdown();
    render.shutdown();
    interaction.shutdown();
    frameSync.shutdown();
    scene.shutdown();
    core.shutdown();
    windowRuntimeState = nullptr;
}

bool RuntimeSystems::isSimulationActive() const {
    const SimGraphRuntime* simGraph = compute.simGraphRuntime();
    return simGraph != nullptr && simGraph->hasNodeInState(SimNodeState::Running);
}

bool RuntimeSystems::isSimulationPaused() const {
    const SimGraphRuntime* simGraph = compute.simGraphRuntime();
    return simGraph != nullptr && simGraph->hasNodeInState(SimNodeState::Paused);
}

uint32_t RuntimeSystems::getSimulationRecordedTimelineFrames() const {
    const SimGraphRuntime* simGraph = compute.simGraphRuntime();
    return simGraph != nullptr ? simGraph->getSimulationRecordedTimelineFrames() : 0;
}

bool RuntimeSystems::isTimelinePlaying() const {
    return timelineRuntime.isPlaying();
}

uint32_t RuntimeSystems::getTimelineCurrentFrame() const {
    return timelineRuntime.getCurrentFrame();
}

uint32_t RuntimeSystems::getTimelineFrameCount() const {
    return timelineRuntime.getFrameCount();
}

uint32_t RuntimeSystems::getTimelineStartDisplayFrame() const {
    return timelineRuntime.getStartDisplayFrame();
}

uint32_t RuntimeSystems::getTimelineEndDisplayFrame() const {
    return timelineRuntime.getEndDisplayFrame();
}

float RuntimeSystems::getTimelineCurrentSeconds() const {
    return timelineRuntime.getCurrentSeconds();
}

float RuntimeSystems::getTimelineDurationSeconds() const {
    return timelineRuntime.getDurationSeconds();
}
