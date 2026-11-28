#pragma once

#include "mesh/MeshElementType.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include "nodegraph/NodeInteraction.hpp"
#include "nodegraph/NodeTransform.hpp"
#include "RuntimeEvent.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

class CameraController;
class InteractionContext;
class ModelRegistry;
class ModelSelection;
class NodeGraphController;
class SceneController;
struct WindowRuntimeState;

struct GizmoDragBegin;
struct GizmoDragUpdate;

class ViewportEditController {
public:
    ViewportEditController(
        InteractionContext& interactionContext,
        NodeGraphController& graphController,
        const SceneController& sceneController,
        ModelRegistry& modelRegistry,
        ModelSelection& modelSelection,
        CameraController& cameraController,
        const WindowRuntimeState& windowState);

    void tick();
    void refreshPickCamera();
    void setDisplayedSourceModelIds(std::vector<uint32_t> modelIds);
    void setSelectedNodes(const std::vector<NodeGraphNodeId>& nodeIds, int primaryNodeId);
    int selectedNode() const { return primarySelectedNode; }
    void reconcileSelection();
    void invalidateSelection();
    bool beginAction(NodeGraphNodeId nodeId, uint64_t nodeInstanceId, uint32_t actionId);
    bool beginViewportSelection();
    bool selectionCommitRequired() const;
    void completeAction();
    void cancelAction();
    void clearActionSelection();
    void setSubselectionElementType(int elementType);
    bool takePendingNodeParameters(NodeGraphNodeId& outNodeId, uint64_t& outOriginInstanceId, std::vector<NodeGraphParamValue>& outParameters);
    bool takePendingActionOutcome(NodeActionCompleted& outOutcome);
    bool takeSelectionRequest(RuntimeNodeSelectionRequested& outRequest);

private:
    enum class SelectionOwnerType : uint8_t {
        None,
        NodeAction,
        Viewport
    };

    struct PendingNodeParameters {
        NodeGraphNodeId nodeId{};
        uint64_t originInstanceId = 0;
        std::vector<NodeGraphParamValue> parameters;
    };

    void queueNodeParameters(
        NodeGraphNodeId nodeId,
        uint64_t originInstanceId,
        std::vector<NodeGraphParamValue> parameters);
    void resetSelectionOwner();

    bool beginGizmoDrag(const GizmoDragBegin& begin);
    void updateGizmoDrag(const GizmoDragUpdate& update);
    void endGizmoDrag();
    void resetGizmoDrag();
    void syncHeatProbeMarkers();
    void processViewportPicks();
    void processSubselectionEditing();
    void refreshInteractionPreview();
    bool beginMeshInteraction(const NodeInteractionRequest& request, const NodeInteractionSession& session);

    InteractionContext& interaction;
    NodeGraphController& graphController;
    const SceneController& sceneController;
    ModelRegistry& modelRegistry;
    ModelSelection& modelSelection;
    CameraController& cameraController;
    const WindowRuntimeState& windowState;

    uint32_t activeRuntimeModelId = 0;
    glm::mat4 dragStartWorld{1.0f};
    std::optional<NodeTransformEditState> activeTransformEdit;
    NodeGraphNodeId activeTransformNodeId{};
    uint64_t activeTransformInstanceId = 0;
    uint64_t nextInteractionSessionId = 1;
    std::vector<NodeGraphNodeId> selectedNodes;
    int primarySelectedNode = 0;
    bool selectionDirty = false;
    uint64_t appliedSelectionRevision = UINT64_MAX;
    std::vector<uint32_t> appliedSelectedRuntimeModelIds;
    uint64_t lastPreviewAuthoredRevision = UINT64_MAX;
    uint64_t lastPreviewEvaluationRevision = UINT64_MAX;
    MeshElementType requestedElementType = MeshElementType::Face;
    SelectionOwnerType selectionOwner = SelectionOwnerType::None;
    NodeInteractionSession activeSession{};
    std::deque<PendingNodeParameters> pendingNodeParameters;
    std::deque<NodeActionCompleted> pendingActionOutcomes;
    std::deque<RuntimeNodeSelectionRequested> pendingSelectionRequests;
};
