#include "ViewportEditController.hpp"

#include "interaction/InteractionContext.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeHeatProbeParams.hpp"
#include "nodegraph/NodeTransform.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/SceneController.hpp"
#include "util/Units.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <optional>
#include <utility>
#include <vector>

ViewportEditController::ViewportEditController(
    InteractionContext& interactionContext,
    const NodeGraphController& graphController,
    const SceneController& sceneController,
    ModelRegistry& modelRegistry)
    : interaction(interactionContext),
      graphController(graphController),
      sceneController(sceneController),
      modelRegistry(modelRegistry) {
}

void ViewportEditController::tick() {
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

    syncHeatProbeMarkers();

    auto& markerInteractionController = interaction.markerInteractionController();
    if (std::optional<MarkerCommit> hit = markerInteractionController.takePendingCommit()) {
        const auto& nodes = graphController.graphState().nodes;
        auto nodeIt = nodes.find(hit->nodeId);
        if (nodeIt != nodes.end()) {
            HeatProbeNodeParams probeParams = readHeatProbeNodeParams(nodeIt->second);
            if (hit->probeIndex >= 0 && hit->probeIndex < static_cast<int>(probeParams.items.size())) {
                probeParams.items[hit->probeIndex].worldPosition = hit->position;
            } else if (!probeParams.items.empty()) {
                probeParams.items[0].worldPosition = hit->position;
            }
            const std::string probesJson = serializeHeatProbesJson(probeParams.items);
            pendingNodeParameters.push_back({
                NodeGraphNodeId{hit->nodeId},
                {
                    {nodegraphparams::heatprobe::ProbesJson, NodeGraphParamType::String, 0.0, 0, false, probesJson}
                }
            });
        }
    }
}

bool ViewportEditController::takePendingNodeParameters(
    NodeGraphNodeId& outNodeId,
    std::vector<NodeGraphParamValue>& outParameters) {
    if (pendingNodeParameters.empty()) {
        return false;
    }

    PendingNodeParameters pending = std::move(pendingNodeParameters.front());
    pendingNodeParameters.pop_front();
    outNodeId = pending.nodeId;
    outParameters = std::move(pending.parameters);
    return true;
}

void ViewportEditController::syncHeatProbeMarkers() {
    const auto& nodes = graphController.graphState().nodes;
    std::vector<MarkerItem> activeMarkers;
    for (const auto& [nodeId, nodeData] : nodes) {
        if (nodeData.typeId == nodegraphtypes::HeatProbe) {
            HeatProbeNodeParams params = readHeatProbeNodeParams(nodeData);
            if (!params.enabled) continue;
            for (int i = 0; i < static_cast<int>(params.items.size()); ++i) {
                const auto& item = params.items[i];
                if (!item.enabled) continue;
                activeMarkers.push_back(MarkerItem{ nodeId, i, item.id, item.worldPosition });
            }
        }
    }
    interaction.markerInteractionController().updateMarkers(activeMarkers);
}

bool ViewportEditController::beginGizmoDrag(const GizmoDragBegin& begin) {
    uint64_t outputSocketKey = 0;
    if (begin.runtimeModelId == 0 ||
        !sceneController.tryGetRuntimeModelSocketKey(begin.runtimeModelId, outputSocketKey) ||
        outputSocketKey == 0) {
        return false;
    }

    NodeGraphNodeId transformNodeId{};
    if (!graphController.resolveGizmoTransformNode(outputSocketKey, transformNodeId)) {
        return false;
    }

    const NodeGraphNode* transformNode = graphController.graphState().node(transformNodeId);
    if (!transformNode) {
        return false;
    }

    glm::mat4 currentWorld{1.0f};
    if (!modelRegistry.tryGetModelMatrix(begin.runtimeModelId, currentWorld)) {
        return false;
    }

    activeTransformNodeId = transformNodeId;
    activeRuntimeModelId = begin.runtimeModelId;
    activeMode = begin.mode;
    activeAxis = begin.axis;
    dragStartParams = readTransformNodeParams(*transformNode);
    previewParams = dragStartParams;
    canonicalToWorldScale = units::canonicalToWorldScale(graphController.worldUnit());

    glm::mat4 initialLocal = NodeTransform::buildLocalTransform(dragStartParams);
    initialLocal[3][0] *= canonicalToWorldScale;
    initialLocal[3][1] *= canonicalToWorldScale;
    initialLocal[3][2] *= canonicalToWorldScale;
    dragParentWorld = currentWorld * glm::inverse(initialLocal);
    return true;
}

void ViewportEditController::updateGizmoDrag(const GizmoDragUpdate& update) {
    if (!activeTransformNodeId.isValid() || activeRuntimeModelId == 0) {
        return;
    }

    previewParams = dragStartParams;
    if (activeMode == GizmoMode::Translate) {
        const glm::vec3 canonicalDelta =
            update.translationDeltaWorld / canonicalToWorldScale;
        previewParams.translateX += canonicalDelta.x;
        previewParams.translateY += canonicalDelta.y;
        previewParams.translateZ += canonicalDelta.z;
    } else if (activeMode == GizmoMode::Rotate) {
        if (activeAxis == GizmoAxis::X) {
            previewParams.rotateXDegrees += update.rotationDeltaDegrees;
        } else if (activeAxis == GizmoAxis::Y) {
            previewParams.rotateYDegrees += update.rotationDeltaDegrees;
        } else if (activeAxis == GizmoAxis::Z) {
            previewParams.rotateZDegrees += update.rotationDeltaDegrees;
        }
    }

    glm::mat4 previewLocal = NodeTransform::buildLocalTransform(previewParams);
    previewLocal[3][0] *= canonicalToWorldScale;
    previewLocal[3][1] *= canonicalToWorldScale;
    previewLocal[3][2] *= canonicalToWorldScale;
    modelRegistry.setModelMatrix(
        activeRuntimeModelId,
        dragParentWorld * previewLocal);
}

void ViewportEditController::endGizmoDrag() {
    if (!activeTransformNodeId.isValid()) {
        resetGizmoDrag();
        return;
    }

    if (activeMode == GizmoMode::Translate) {
        pendingNodeParameters.push_back({
            activeTransformNodeId,
            {
                {nodegraphparams::transform::TranslateX, NodeGraphParamType::Float, previewParams.translateX},
                {nodegraphparams::transform::TranslateY, NodeGraphParamType::Float, previewParams.translateY},
                {nodegraphparams::transform::TranslateZ, NodeGraphParamType::Float, previewParams.translateZ}
            }
        });
    } else if (activeMode == GizmoMode::Rotate) {
        pendingNodeParameters.push_back({
            activeTransformNodeId,
            {
                {nodegraphparams::transform::RotateXDegrees, NodeGraphParamType::Float, previewParams.rotateXDegrees},
                {nodegraphparams::transform::RotateYDegrees, NodeGraphParamType::Float, previewParams.rotateYDegrees},
                {nodegraphparams::transform::RotateZDegrees, NodeGraphParamType::Float, previewParams.rotateZDegrees}
            }
        });
    }

    resetGizmoDrag();
}

void ViewportEditController::resetGizmoDrag() {
    activeTransformNodeId = {};
    activeRuntimeModelId = 0;
    dragStartParams = {};
    previewParams = {};
    dragParentWorld = glm::mat4(1.0f);
    canonicalToWorldScale = 1.0f;
    activeMode = GizmoMode::Translate;
    activeAxis = GizmoAxis::None;
}