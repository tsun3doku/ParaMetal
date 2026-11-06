#include "MarkerInteractionController.hpp"

#include "Camera.hpp"
#include "MousePicker.hpp"
#include "render/WindowRuntimeState.hpp"
#include "scene/Model.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <algorithm>
#include <cmath>

MarkerInteractionController::MarkerInteractionController(
    Camera& camera,
    ModelRegistry& modelRegistry,
    const WindowRuntimeState& windowState)
    : camera(camera),
      modelRegistry(modelRegistry),
      windowState(windowState) {
}

void MarkerInteractionController::updateMarkers(const std::vector<MarkerItem>& updatedMarkers) {
    if (!isDragging && !pendingCommit.has_value()) {
        markers = updatedMarkers;
        return;
    }

    // While dragging or while commit is pending, preserve the live position of the active dragged marker
    std::vector<MarkerItem> next = updatedMarkers;
    for (auto& item : next) {
        if (item.nodeId == selectedNodeId && item.probeIndex == selectedProbeIndex) {
            item.worldPosition = dragCurrentPosition;
        }
    }
    markers = std::move(next);
}

void MarkerInteractionController::setSelected(uint32_t nodeId, int probeIndex, uint32_t probeId) {
    selectedNodeId = nodeId;
    selectedProbeIndex = probeIndex;
    selectedProbeId = probeId;
}

void MarkerInteractionController::clearSelection() {
    selectedNodeId = 0;
    selectedProbeIndex = -1;
    selectedProbeId = 0;
}

void MarkerInteractionController::startProbePicking(uint32_t targetNodeId, int probeIndex) {
    probePickingActive = true;
    probePickingNodeId = targetNodeId;
    probePickingIndex = probeIndex;
    pendingCommit.reset();
}

void MarkerInteractionController::cancelProbePicking() {
    probePickingActive = false;
    probePickingNodeId = 0;
    probePickingIndex = 0;
    pendingCommit.reset();
}

std::optional<MarkerCommit> MarkerInteractionController::takePendingCommit() {
    if (pendingCommit) {
        auto commit = pendingCommit;
        pendingCommit.reset();
        return commit;
    }
    return std::nullopt;
}

std::optional<uint32_t> MarkerInteractionController::takePendingSelectedNodeId() {
    if (pendingSelectedNodeId) {
        auto id = pendingSelectedNodeId;
        pendingSelectedNodeId.reset();
        return id;
    }
    return std::nullopt;
}

int MarkerInteractionController::hitTest(float mouseX, float mouseY) const {
    if (markers.empty()) return -1;

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) return -1;

    int closestIdx = -1;
    float closestZ = 1e9f;

    for (size_t i = 0; i < markers.size(); ++i) {
        float depth = 0.0f;
        const auto screenPos = camera.worldToScreen(markers[i].worldPosition, width, height, &depth);
        if (!screenPos) continue;

        // needle tip hit bounds
        if (mouseX >= (screenPos->x - 24.0f) && mouseX <= (screenPos->x + 24.0f) &&
            mouseY >= (screenPos->y - 44.0f) && mouseY <= (screenPos->y + 8.0f)) {
            if (depth < closestZ) {
                closestZ = depth;
                closestIdx = static_cast<int>(i);
            }
        }
    }

    return closestIdx;
}

bool MarkerInteractionController::handlePointerPress(float mouseX, float mouseY, int button, bool shiftPressed) {
    (void)shiftPressed;
    if (button == 2) {
        if (probePickingActive) {
            cancelProbePicking();
            return true;
        }
        return false;
    }
    if (button != 1) {
        return false;
    }

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) return false;

    // 1. Initial surface placement mode ("Pick surface coordinate on 3D model")
    if (probePickingActive) {
        const Ray ray = MousePicker::screenToWorldRay(mouseX, mouseY, width, height, camera);
        RayHit closestHit{};
        bool hitAny = false;
        for (uint32_t modelId : modelRegistry.getRenderableModelIds()) {
            const Model* model = modelRegistry.getModel(modelId);
            if (!model) continue;
            RayHit hit{};
            if (MousePicker::raycastModel(ray, *model, hit)) {
                if (!hitAny || hit.distance < closestHit.distance) {
                    closestHit = hit;
                    hitAny = true;
                }
            }
        }

        if (hitAny) {
            pendingCommit = MarkerCommit{ probePickingNodeId, probePickingIndex, closestHit.worldPosition };
            probePickingActive = false;
            probePickingNodeId = 0;
            probePickingIndex = 0;
        }
        return true;
    }

    // 2. Direct viewport marker selection and dragging
    const int hitIdx = hitTest(mouseX, mouseY);
    if (hitIdx >= 0 && hitIdx < static_cast<int>(markers.size())) {
        const auto& hitMarker = markers[hitIdx];
        selectedNodeId = hitMarker.nodeId;
        selectedProbeIndex = hitMarker.probeIndex;
        selectedProbeId = hitMarker.probeId;
        pendingSelectedNodeId = hitMarker.nodeId;

        isDragging = true;
        dragCurrentPosition = hitMarker.worldPosition;
        dragStartPosition = hitMarker.worldPosition;
        return true;
    }

    return false;
}

bool MarkerInteractionController::handlePointerMove(float mouseX, float mouseY) {
    if (!isDragging) {
        return false;
    }

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) return false;

    const Ray ray = MousePicker::screenToWorldRay(mouseX, mouseY, width, height, camera);
    RayHit closestHit{};
    bool hitAny = false;

    for (uint32_t modelId : modelRegistry.getRenderableModelIds()) {
        const Model* model = modelRegistry.getModel(modelId);
        if (!model) continue;
        RayHit hit{};
        if (MousePicker::raycastModel(ray, *model, hit)) {
            if (!hitAny || hit.distance < closestHit.distance) {
                closestHit = hit;
                hitAny = true;
            }
        }
    }

    if (hitAny) {
        dragCurrentPosition = closestHit.worldPosition;
        for (auto& item : markers) {
            if (item.nodeId == selectedNodeId && item.probeIndex == selectedProbeIndex) {
                item.worldPosition = dragCurrentPosition;
                break;
            }
        }
    }

    return true;
}

bool MarkerInteractionController::handlePointerRelease(float mouseX, float mouseY, int button) {
    (void)mouseX;
    (void)mouseY;
    if (button != 1) {
        return false;
    }

    if (isDragging) {
        if (glm::distance(dragCurrentPosition, dragStartPosition) > 0.001f) {
            pendingCommit = MarkerCommit{ selectedNodeId, selectedProbeIndex, dragCurrentPosition };
        }
        isDragging = false;
        return true;
    }

    return false;
}
