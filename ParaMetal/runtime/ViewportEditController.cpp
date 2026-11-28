#include "ViewportEditController.hpp"

#include "interaction/InteractionContext.hpp"
#include "mesh/SubselectionEditState.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeHeatProbe.hpp"
#include "render/WindowRuntimeState.hpp"
#include "scene/Camera.hpp"
#include "scene/CameraController.hpp"
#include "scene/GizmoController.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/Model.hpp"
#include "scene/ModelSelection.hpp"
#include "scene/SceneController.hpp"
#include "scene/SubselectionInteractionController.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

ViewportEditController::ViewportEditController(
    InteractionContext& interactionContext,
    NodeGraphController& graphController,
    const SceneController& sceneController,
    ModelRegistry& modelRegistry,
    ModelSelection& modelSelection,
    CameraController& cameraController,
    const WindowRuntimeState& windowState)
    : interaction(interactionContext),
      graphController(graphController),
      sceneController(sceneController),
      modelRegistry(modelRegistry),
      modelSelection(modelSelection),
      cameraController(cameraController),
      windowState(windowState) {
}

void ViewportEditController::tick() {
    processViewportPicks();

    auto& gizmoController = interaction.gizmoController();

    if (std::optional<GizmoDragBegin> begin = gizmoController.takePendingDragBegin()) {
        if (!beginGizmoDrag(*begin)) {
            gizmoController.cancelDrag();
            resetGizmoDrag();
        }
    }

    if (std::optional<GizmoDragUpdate> update = gizmoController.takePendingDragUpdate()) {
        updateGizmoDrag(*update);
    }

    if (gizmoController.takePendingDragEnd()) {
        endGizmoDrag();
    }
    if (activeRuntimeModelId != 0 && !gizmoController.isActive()) {
        resetGizmoDrag();
    }

    syncHeatProbeMarkers();

    auto& markerInteractionController = interaction.markerInteractionController();
    if (std::optional<MarkerCommit> hit = markerInteractionController.takePendingCommit()) {
        std::vector<NodeGraphParamValue> parameters;
        uint64_t originInstanceId = 0;
        if (buildHeatProbePositionParameters(
                graphController.graphState(),
                hit->nodeId,
                hit->probeIndex,
                hit->position,
                originInstanceId,
                parameters)) {
            queueNodeParameters(
                NodeGraphNodeId{hit->nodeId},
                originInstanceId,
                std::move(parameters));
        }
    }

    processSubselectionEditing();
    SubselectionInteractionController& subselection = interaction.subselectionInteractionController();
    subselection.tick();
    if (selectionOwner == SelectionOwnerType::NodeAction) {
        refreshInteractionPreview();
    } else if (selectionOwner == SelectionOwnerType::Viewport) {
        subselection.setDisplaySelection(subselection.authoredSelection());
    }
}

void ViewportEditController::refreshPickCamera() {
    SubselectionInteractionController& subselection = interaction.subselectionInteractionController();
    if (!subselection.isActive()) {
        return;
    }

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) {
        return;
    }

    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    subselection.setViewProjection(
        cameraController.getCamera().getProjectionMatrix(aspect) *
        cameraController.getCamera().getViewMatrix(),
        glm::vec2(static_cast<float>(width), static_cast<float>(height)));
}

void ViewportEditController::processViewportPicks() {
    const uint64_t graphInstanceId = graphController.graphState().instanceId;

    for (const ModelSelectionPick& pick : modelSelection.takePendingPicks()) {
        if (pick.result.isGizmo()) {
            continue;
        }

        if (pick.result.isModel()) {
            RuntimePackageController::DisplayedSurface surface{};
            if (!graphController.getPackageController().findDisplayedSurface(
                    pick.result.modelID, surface)) {
                continue;
            }

            NodeGraphNodeId nodeId{};
            if (!graphController.primaryDisplayNode(nodeId)) {
                continue;
            }
            const NodeGraphNode* node =
                graphController.graphState().node(nodeId);
            if (!node) {
                continue;
            }

            pendingSelectionRequests.push_back({
                graphInstanceId,
                nodeId,
                node->instanceId,
                pick.shiftPressed,
            });
        } else if (!pick.shiftPressed) {
            pendingSelectionRequests.push_back({
                graphInstanceId,
                {},
                0,
                false,
            });
        }
    }
}

void ViewportEditController::setSelectedNodes(
    const std::vector<NodeGraphNodeId>& nodeIds,
    int primaryNodeId) {
    selectedNodes.clear();
    selectedNodes.reserve(nodeIds.size());
    for (NodeGraphNodeId nodeId : nodeIds) {
        if (nodeId.value == 0) {
            continue;
        }
        if (std::find(selectedNodes.begin(), selectedNodes.end(), nodeId) ==
            selectedNodes.end()) {
            selectedNodes.push_back(nodeId);
        }
    }

    primarySelectedNode = std::max(0, primaryNodeId);
    if (primarySelectedNode != 0 &&
        std::find(
            selectedNodes.begin(),
            selectedNodes.end(),
            NodeGraphNodeId{static_cast<uint32_t>(primarySelectedNode)}) ==
            selectedNodes.end()) {
        selectedNodes.push_back(
            NodeGraphNodeId{static_cast<uint32_t>(primarySelectedNode)});
    }
    if (primarySelectedNode == 0 && !selectedNodes.empty()) {
        primarySelectedNode = static_cast<int>(selectedNodes.back().value);
    }
    selectionDirty = true;
}

void ViewportEditController::reconcileSelection() {
    const uint64_t revision = graphController.runtimeModelRevision();
    if (!selectionDirty && revision == appliedSelectionRevision) {
        return;
    }
    if (!selectionDirty &&
        selectedNodes.empty() &&
        appliedSelectedRuntimeModelIds.empty()) {
        appliedSelectionRevision = revision;
        return;
    }
    const bool forceApply = selectionDirty;
    selectionDirty = false;
    appliedSelectionRevision = revision;

    std::vector<uint32_t> resolvedIds;
    for (NodeGraphNodeId nodeId : selectedNodes) {
        std::vector<uint32_t> nodeModelIds;
        if (!graphController.runtimeModelIdsForNode(nodeId, nodeModelIds)) {
            continue;
        }
        for (uint32_t runtimeModelId : nodeModelIds) {
            if (std::find(
                    resolvedIds.begin(),
                    resolvedIds.end(),
                    runtimeModelId) == resolvedIds.end()) {
                resolvedIds.push_back(runtimeModelId);
            }
        }
    }
    if (!forceApply && resolvedIds == appliedSelectedRuntimeModelIds) {
        return;
    }

    appliedSelectedRuntimeModelIds = resolvedIds;
    modelSelection.clearSelection();
    for (uint32_t runtimeModelId : appliedSelectedRuntimeModelIds) {
        modelSelection.addSelectedModelID(runtimeModelId);
    }
}

void ViewportEditController::invalidateSelection() {
    resetGizmoDrag();
    if (interaction.subselectionInteractionController().isActive()) {
        interaction.subselectionInteractionController().end();
    }
    resetSelectionOwner();
    selectedNodes.clear();
    primarySelectedNode = 0;
    pendingSelectionRequests.clear();
    modelSelection.discardPendingPicks();
    modelSelection.clearSelection();
    appliedSelectedRuntimeModelIds.clear();
    selectionDirty = true;
    appliedSelectionRevision = UINT64_MAX;
}

void ViewportEditController::resetSelectionOwner() {
    selectionOwner = SelectionOwnerType::None;
    activeSession = {};
    lastPreviewAuthoredRevision = UINT64_MAX;
    lastPreviewEvaluationRevision = UINT64_MAX;
}

bool ViewportEditController::beginAction(
    NodeGraphNodeId nodeId,
    uint64_t nodeInstanceId,
    uint32_t actionId) {
    if (selectionOwner != SelectionOwnerType::None) {
        cancelAction();
    }

    NodeInteractionRequest request{};
    NodeInteractionSession session{};
    std::string error;
    if (!graphController.beginAction(
            nodeId,
            nodeInstanceId,
            actionId,
            request,
            session,
            error)) {
        pendingActionOutcomes.push_back({nodeId, false, std::move(error)});
        return false;
    }
    session.id = nextInteractionSessionId++;
    if (session.id == 0) {
        session.id = nextInteractionSessionId++;
    }

    bool begun = false;
    switch (request.type) {
    case NodeInteractionType::MeshSubselection:
        begun = beginMeshInteraction(request, session);
        break;
    case NodeInteractionType::Transform:
    case NodeInteractionType::Point:
    case NodeInteractionType::Face:
        break;
    }

    if (!begun) {
        pendingActionOutcomes.push_back({
            nodeId,
            false,
            "Could not begin node interaction.",
        });
        return false;
    }

    activeSession = session;
    selectionOwner = SelectionOwnerType::NodeAction;
    lastPreviewAuthoredRevision = UINT64_MAX;
    lastPreviewEvaluationRevision = UINT64_MAX;
    refreshInteractionPreview();
    return selectionOwner == SelectionOwnerType::NodeAction;
}

bool ViewportEditController::beginViewportSelection() {
    if (selectionOwner != SelectionOwnerType::None) {
        cancelAction();
    }

    const std::vector<uint32_t>& selected = modelSelection.getSelectedModelIDsRenderThread();
    if (selected.size() != 1) {
        return false;
    }

    const uint32_t runtimeModelId = selected.front();
    RuntimePackageController::DisplayedSurface surface{};
    if (!graphController.getPackageController().findDisplayedSurface(runtimeModelId, surface)) {
        return false;
    }

    NodeInteractionSession session{};
    session.type = NodeInteractionType::MeshSubselection;
    session.binding = surface.intrinsic
        ? SurfaceBinding::Intrinsic
        : SurfaceBinding::Source;
    session.sourceSocketKey = surface.socketKey;
    session.displaySocketKey = surface.socketKey;
    session.sourceSurfaceIdentity = surface.surfaceIdentity;
    session.runtimeModelId = runtimeModelId;

    NodeInteractionRequest request{};
    MeshSelection emptySelection{};
    emptySelection.elementType = requestedElementType;
    request.initialValue = std::move(emptySelection);

    if (!beginMeshInteraction(request, session)) {
        return false;
    }

    activeSession = session;
    selectionOwner = SelectionOwnerType::Viewport;
    return true;
}

bool ViewportEditController::selectionCommitRequired() const {
    return selectionOwner == SelectionOwnerType::NodeAction &&
        activeSession.type == NodeInteractionType::MeshSubselection &&
        interaction.subselectionInteractionController().isActive();
}

void ViewportEditController::completeAction() {
    if (selectionOwner != SelectionOwnerType::NodeAction) {
        return;
    }

    SubselectionInteractionController& controller =
        interaction.subselectionInteractionController();
    if (activeSession.type != NodeInteractionType::MeshSubselection ||
        !controller.isActive()) {
        cancelAction();
        return;
    }

    const NodeInteractionSession session = activeSession;
    MeshSelection selection = controller.authoredSelection();

    NodeInteractionResult result{};
    result.sessionId = session.id;
    result.value = std::move(selection);

    std::vector<NodeGraphParamValue> patch;
    std::string error;
    const bool completed = graphController.completeAction(session, result, patch, error);

    controller.end();
    resetSelectionOwner();

    if (completed) {
        queueNodeParameters(
            session.nodeId, session.nodeInstanceId, std::move(patch));
    } else {
        pendingActionOutcomes.push_back({
            session.nodeId,
            false,
            std::move(error),
        });
    }
}

void ViewportEditController::cancelAction() {
    SubselectionInteractionController& controller = interaction.subselectionInteractionController();
    if (controller.isActive()) {
        controller.end();
    }
    resetSelectionOwner();
}

void ViewportEditController::clearActionSelection() {
    SubselectionInteractionController& controller = interaction.subselectionInteractionController();
    if (selectionOwner != SelectionOwnerType::None &&
        controller.isActive()) {
        controller.clearSelection();
    }
}

void ViewportEditController::setSubselectionElementType(int elementType) {
    if (elementType < static_cast<int>(MeshElementType::Vertex) ||
        elementType > static_cast<int>(MeshElementType::Face)) {
        return;
    }

    requestedElementType = static_cast<MeshElementType>(elementType);

    interaction.subselectionInteractionController()
        .changeElementType(requestedElementType);
}

void ViewportEditController::setDisplayedSourceModelIds(
    std::vector<uint32_t> modelIds) {
    interaction.markerInteractionController().updateDisplayedModelIds(std::move(modelIds));
}

void ViewportEditController::processSubselectionEditing() {
    SubselectionInteractionController& controller = interaction.subselectionInteractionController();
    if (!controller.isActive()) {
        return;
    }

    const SubselectionEditState& state = controller.editState();
    const std::vector<uint32_t>& selectedModels =
        modelSelection.getSelectedModelIDsRenderThread();
    glm::mat4 modelMatrix;
    if (selectionOwner == SelectionOwnerType::None ||
        !graphController.meshSelectionIsCurrent(state) ||
        (selectionOwner == SelectionOwnerType::Viewport &&
            (selectedModels.size() != 1 ||
             selectedModels.front() != state.runtimeModelId)) ||
        !modelRegistry.tryGetModelMatrix(state.runtimeModelId, modelMatrix)) {
        if (selectionOwner == SelectionOwnerType::NodeAction) {
            pendingActionOutcomes.push_back({
                activeSession.nodeId,
                false,
                "Interaction source changed.",
            });
        }
        controller.end();
        resetSelectionOwner();
    } else {
        controller.setModelMatrix(modelMatrix);
    }
}

void ViewportEditController::refreshInteractionPreview() {
    if (selectionOwner != SelectionOwnerType::NodeAction ||
        activeSession.type != NodeInteractionType::MeshSubselection) {
        return;
    }

    SubselectionInteractionController& subselection =
        interaction.subselectionInteractionController();
    if (!subselection.isActive()) {
        return;
    }

    const uint64_t authoredRevision = subselection.selectionRevision();
    const uint64_t evaluationRevision = graphController.evaluationRevision();
    if (authoredRevision == lastPreviewAuthoredRevision &&
        evaluationRevision == lastPreviewEvaluationRevision) {
        return;
    }

    MeshSelection authored = subselection.authoredSelection();

    NodeInteractionValue display;
    std::string error;
    const bool succeeded = graphController.previewAction(activeSession, authored, display, error);
    const MeshSelection* preview = succeeded
        ? std::get_if<MeshSelection>(&display)
        : nullptr;

    if (!preview || !subselection.setDisplaySelection(*preview)) {
        pendingActionOutcomes.push_back({
            activeSession.nodeId,
            false,
            error.empty() ? "Invalid interaction preview." : std::move(error),
        });
        cancelAction();
        return;
    }

    lastPreviewAuthoredRevision = authoredRevision;
    lastPreviewEvaluationRevision = evaluationRevision;
}

bool ViewportEditController::beginMeshInteraction(
    const NodeInteractionRequest& request,
    const NodeInteractionSession& session) {
    const MeshSelection* initialSelection = std::get_if<MeshSelection>(&request.initialValue);
    if (!initialSelection) {
        return false;
    }

    SubselectionEditState state{};
    if (!graphController.beginMeshSelection(session, state)) {
        return false;
    }
    if (!modelRegistry.tryGetModelMatrix(session.runtimeModelId, state.modelMatrix)) {
        return false;
    }

    const Model* model = modelRegistry.getModel(session.runtimeModelId);
    if (!model) {
        return false;
    }
    state.pickGeometryIdentity =
        session.binding == SurfaceBinding::Intrinsic
            ? state.surfaceIdentity
            : model->getGeometryIdentity();

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) {
        return false;
    }
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    state.viewProjection = cameraController.getCamera().getProjectionMatrix(aspect) * cameraController.getCamera().getViewMatrix();
    state.viewportSize = glm::vec2(static_cast<float>(width), static_cast<float>(height));

    SubselectionInteractionController& controller = interaction.subselectionInteractionController();
    controller.begin(std::move(state));
    controller.setSelection(*initialSelection);
    controller.setDisplaySelection(*initialSelection);
    return true;
}
void ViewportEditController::queueNodeParameters(
    NodeGraphNodeId nodeId,
    uint64_t originInstanceId,
    std::vector<NodeGraphParamValue> parameters) {
    pendingNodeParameters.push_back({
        nodeId,
        originInstanceId,
        std::move(parameters)
    });
}

bool ViewportEditController::takePendingNodeParameters(
    NodeGraphNodeId& outNodeId,
    uint64_t& outOriginInstanceId,
    std::vector<NodeGraphParamValue>& outParameters) {
    if (pendingNodeParameters.empty()) {
        return false;
    }

    PendingNodeParameters pending = std::move(pendingNodeParameters.front());
    pendingNodeParameters.pop_front();
    outNodeId = pending.nodeId;
    outOriginInstanceId = pending.originInstanceId;
    outParameters = std::move(pending.parameters);
    return true;
}

bool ViewportEditController::takePendingActionOutcome(NodeActionCompleted& outOutcome) {
    if (pendingActionOutcomes.empty()) {
        return false;
    }
    outOutcome = std::move(pendingActionOutcomes.front());
    pendingActionOutcomes.pop_front();
    return true;
}

bool ViewportEditController::takeSelectionRequest(
    RuntimeNodeSelectionRequested& outRequest) {
    if (pendingSelectionRequests.empty()) {
        return false;
    }
    outRequest = pendingSelectionRequests.front();
    pendingSelectionRequests.pop_front();
    return true;
}

void ViewportEditController::syncHeatProbeMarkers() {
    const std::vector<NodeHeatProbeMarker> markers = collectHeatProbeMarkers(graphController.graphState());
    std::vector<MarkerItem> activeMarkers;
    activeMarkers.reserve(markers.size());
    for (const NodeHeatProbeMarker& marker : markers) {
        activeMarkers.push_back(MarkerItem{
            marker.nodeId,
            marker.probeIndex,
            marker.probeId,
            marker.worldPosition});
    }
    interaction.markerInteractionController().updateMarkers(activeMarkers);
}

bool ViewportEditController::beginGizmoDrag(const GizmoDragBegin& begin) {
    uint64_t outputSocketKey = 0;
    if (begin.runtimeModelId == 0 ||
        !sceneController.tryGetRuntimeModelSocketKey(
            begin.runtimeModelId, outputSocketKey) ||
        outputSocketKey == 0) {
        return false;
    }

    glm::mat4 currentWorld{1.0f};
    if (!modelRegistry.tryGetModelMatrix(begin.runtimeModelId, currentWorld)) {
        return false;
    }

    NodeGraphNodeId transformNodeId{};
    if (!graphController.resolveGizmoTransformNode(outputSocketKey, transformNodeId)) {
        return false;
    }
    const NodeGraphNode* transformNode =
        graphController.graphState().node(transformNodeId);
    if (!transformNode) {
        return false;
    }

    NodeTransformEditState state{};
    if (!NodeTransform::beginEdit(
            *transformNode,
            currentWorld,
            graphController.worldUnit(),
            static_cast<int>(begin.mode),
            static_cast<int>(begin.axis),
            state)) {
        return false;
    }

    activeTransformEdit = std::move(state);
    activeTransformNodeId = transformNodeId;
    activeTransformInstanceId = transformNode->instanceId;
    activeRuntimeModelId = begin.runtimeModelId;
    dragStartWorld = currentWorld;
    return true;
}

void ViewportEditController::updateGizmoDrag(const GizmoDragUpdate& update) {
    if (activeRuntimeModelId == 0 || !activeTransformEdit.has_value()) {
        return;
    }

    glm::mat4 previewWorld{1.0f};
    if (!NodeTransform::previewEdit(
            *activeTransformEdit,
            update.translationDeltaWorld,
            update.rotationDeltaDegrees,
            previewWorld)) {
        return;
    }
    modelRegistry.setModelMatrix(activeRuntimeModelId, previewWorld);
}

void ViewportEditController::endGizmoDrag() {
    if (activeRuntimeModelId == 0 || !activeTransformEdit.has_value()) {
        resetGizmoDrag();
        return;
    }

    std::vector<NodeGraphParamValue> parameters;
    if (NodeTransform::buildEditParameters(
            *activeTransformEdit, parameters)) {
        queueNodeParameters(
            activeTransformNodeId,
            activeTransformInstanceId,
            std::move(parameters));
    }
    resetGizmoDrag();
}

void ViewportEditController::resetGizmoDrag() {
    if (activeRuntimeModelId != 0) {
        modelRegistry.setModelMatrix(activeRuntimeModelId, dragStartWorld);
    }
    activeTransformEdit.reset();
    activeTransformNodeId = {};
    activeTransformInstanceId = 0;
    activeRuntimeModelId = 0;
    dragStartWorld = glm::mat4(1.0f);
}
