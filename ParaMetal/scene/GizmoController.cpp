#include "GizmoController.hpp"

#include "Camera.hpp"
#include "CameraController.hpp"
#include "ModelSelection.hpp"
#include "render/WindowRuntimeState.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <cmath>

GizmoController::GizmoController(
    const CameraController& cameraController,
    const WindowRuntimeState& windowState)
    : cameraController(cameraController),
      camera(cameraController.getCamera()),
      windowState(windowState) {
}

bool GizmoController::handlePointerPress(float mouseX, float mouseY, int button, bool shiftPressed) {
    (void)mouseX;
    (void)mouseY;
    (void)button;
    (void)shiftPressed;
    return isDraggingGizmo;
}

bool GizmoController::handlePointerMove(float mouseX, float mouseY) {
    if (!isDraggingGizmo) {
        return false;
    }

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) {
        return true;
    }

    const glm::vec3 rayOrigin = camera.screenToWorldRayOrigin(mouseX, mouseY, width, height);
    const glm::vec3 rayDir = camera.screenToWorldRay(mouseX, mouseY, width, height);

    if (getMode() == GizmoMode::Translate) {
        accumulatedTranslation = calculateTranslationDelta(rayOrigin, rayDir, cachedGizmoPosition, getActiveAxis());
    } else if (getMode() == GizmoMode::Rotate) {
        accumulatedRotation = calculateRotationDelta(rayOrigin, rayDir, cachedGizmoPosition, getActiveAxis());
    }

    pendingGizmoDragUpdate = GizmoDragUpdate{
        accumulatedTranslation,
        accumulatedRotation
    };
    return true;
}

bool GizmoController::handlePointerRelease(float mouseX, float mouseY, int button) {
    (void)mouseX;
    (void)mouseY;
    if (button != 1) {
        return false;
    }

    if (isDraggingGizmo) {
        pendingGizmoDragEnd = true;
        isDraggingGizmo = false;
        endDrag();
        accumulatedTranslation = glm::vec3(0.0f);
        accumulatedRotation = 0.0f;
        return true;
    }

    return false;
}

void GizmoController::updateGizmo(const ModelSelection& modelSelection, ModelRegistry& modelRegistry) {
    if (isDraggingGizmo) {
        return;
    }

    const PickedResult lastPick = modelSelection.getLastPickedResult();
    if (lastPick.isGizmo() && modelSelection.getSelected()) {
        GizmoAxis hitAxis = GizmoAxis::None;
        if (lastPick.gizmoAxis == PickedGizmoAxis::X) {
            hitAxis = GizmoAxis::X;
        } else if (lastPick.gizmoAxis == PickedGizmoAxis::Y) {
            hitAxis = GizmoAxis::Y;
        } else if (lastPick.gizmoAxis == PickedGizmoAxis::Z) {
            hitAxis = GizmoAxis::Z;
        }

        if (hitAxis != GizmoAxis::None) {
            const uint32_t runtimeModelId = modelSelection.getSelectedModelID();
            if (runtimeModelId == 0) {
                const_cast<ModelSelection&>(modelSelection).clearLastPickedResult();
                return;
            }

            if (lastPick.gizmoMode == PickedGizmoMode::Translate) {
                setMode(GizmoMode::Translate);
            } else if (lastPick.gizmoMode == PickedGizmoMode::Rotate) {
                setMode(GizmoMode::Rotate);
            }

            const PickingRequest pickReq = modelSelection.getLastPickRequest();
            const glm::vec3 gizmoPosition = calculateGizmoPosition(modelRegistry, modelSelection);
            const uint32_t width = windowState.width.load(std::memory_order_acquire);
            const uint32_t height = windowState.height.load(std::memory_order_acquire);
            const glm::vec3 rayOrigin = camera.screenToWorldRayOrigin(pickReq.mouseX, pickReq.mouseY, width, height);
            const glm::vec3 rayDir = camera.screenToWorldRay(pickReq.mouseX, pickReq.mouseY, width, height);

            isDraggingGizmo = true;
            cachedGizmoPosition = gizmoPosition;
            startDrag(hitAxis, rayOrigin, rayDir, cachedGizmoPosition);
            accumulatedTranslation = glm::vec3(0.0f);
            accumulatedRotation = 0.0f;
            pendingGizmoDragBegin = GizmoDragBegin{
                runtimeModelId,
                getMode(),
                hitAxis
            };

            const_cast<ModelSelection&>(modelSelection).clearLastPickedResult();
        }
    }
}

std::optional<GizmoDragBegin> GizmoController::takePendingDragBegin() {
    auto pending = std::move(pendingGizmoDragBegin);
    pendingGizmoDragBegin.reset();
    return pending;
}

std::optional<GizmoDragUpdate> GizmoController::takePendingDragUpdate() {
    auto pending = std::move(pendingGizmoDragUpdate);
    pendingGizmoDragUpdate.reset();
    return pending;
}

bool GizmoController::takePendingDragEnd() {
    const bool pending = pendingGizmoDragEnd;
    pendingGizmoDragEnd = false;
    return pending;
}

void GizmoController::cancelDrag() {
    isDraggingGizmo = false;
    endDrag();
    accumulatedTranslation = glm::vec3(0.0f);
    accumulatedRotation = 0.0f;
    pendingGizmoDragBegin.reset();
    pendingGizmoDragUpdate.reset();
    pendingGizmoDragEnd = false;
}

void GizmoController::startDrag(GizmoAxis axis, const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition) {
    activeAxis = axis;
    dragStartPos = gizmoPosition;
    dragStartRayOrigin = rayOrigin;
    dragStartRayDir = rayDir;

    glm::vec3 axisDir(0.0f);
    if (axis == GizmoAxis::X) {
        axisDir = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Y) {
        axisDir = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Z) {
        axisDir = glm::vec3(0.0f, 0.0f, 1.0f);
    }

    const glm::vec3 w0 = gizmoPosition - rayOrigin;
    const float a = glm::dot(axisDir, axisDir);
    const float b = glm::dot(axisDir, rayDir);
    const float c = glm::dot(rayDir, rayDir);
    const float d = glm::dot(axisDir, w0);
    const float e = glm::dot(rayDir, w0);

    const float denom = a * c - b * b;
    if (fabs(denom) < 0.0001f) {
        dragStartIntersection = gizmoPosition;
        return;
    }

    const float sc = (b * e - c * d) / denom;
    dragStartIntersection = gizmoPosition + axisDir * sc;
}

void GizmoController::endDrag() {
    activeAxis = GizmoAxis::None;
}

glm::vec3 GizmoController::calculateTranslationDelta(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition, GizmoAxis axis) {
    (void)gizmoPosition;

    if (axis == GizmoAxis::None) {
        return glm::vec3(0.0f);
    }

    glm::vec3 axisDir(0.0f);
    if (axis == GizmoAxis::X) {
        axisDir = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Y) {
        axisDir = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Z) {
        axisDir = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    else {
        return glm::vec3(0.0f);
    }

    const glm::vec3 w0 = dragStartPos - rayOrigin;
    const float a = glm::dot(axisDir, axisDir);
    const float b = glm::dot(axisDir, rayDir);
    const float c = glm::dot(rayDir, rayDir);
    const float d = glm::dot(axisDir, w0);
    const float e = glm::dot(rayDir, w0);

    const float denom = a * c - b * b;
    if (fabs(denom) < 0.0001f) {
        return glm::vec3(0.0f);
    }

    const float sc = (b * e - c * d) / denom;
    const glm::vec3 currentIntersection = dragStartPos + axisDir * sc;
    const glm::vec3 translation = currentIntersection - dragStartIntersection;

    const float distance = glm::dot(translation, axisDir);
    return axisDir * distance;
}

float GizmoController::calculateRotationDelta(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition, GizmoAxis axis) {
    if (axis == GizmoAxis::None) {
        return 0.0f;
    }

    glm::vec3 axisDir(0.0f);
    if (axis == GizmoAxis::X) {
        axisDir = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Y) {
        axisDir = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    else if (axis == GizmoAxis::Z) {
        axisDir = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    else {
        return 0.0f;
    }

    const float rayDotAxis = glm::dot(rayDir, axisDir);
    if (fabs(rayDotAxis) < 0.0001f) {
        return 0.0f;
    }

    const float startRayDotAxis = glm::dot(dragStartRayDir, axisDir);
    if (fabs(startRayDotAxis) < 0.0001f) {
        return 0.0f;
    }

    const float t = glm::dot(gizmoPosition - rayOrigin, axisDir) / rayDotAxis;
    const glm::vec3 currentPoint = rayOrigin + rayDir * t;

    const float t0 = glm::dot(gizmoPosition - dragStartRayOrigin, axisDir) / startRayDotAxis;
    const glm::vec3 startPoint = dragStartRayOrigin + dragStartRayDir * t0;

    const glm::vec3 startVec = glm::normalize(startPoint - gizmoPosition);
    const glm::vec3 currentVec = glm::normalize(currentPoint - gizmoPosition);

    const float cosAngle = glm::clamp(glm::dot(startVec, currentVec), -1.0f, 1.0f);
    float angle = acos(cosAngle);

    const glm::vec3 cross = glm::cross(startVec, currentVec);
    if (glm::dot(cross, axisDir) < 0.0f) {
        angle = -angle;
    }

    return glm::degrees(angle);
}

glm::vec3 GizmoController::calculateGizmoPosition(ModelRegistry& modelRegistry, const ModelSelection& modelSelection) {
    const auto& selectedModelIDs = modelSelection.getSelectedModelIDsRenderThread();
    glm::vec3 gizmoPosition(0.0f);
    int count = 0;

    for (uint32_t id : selectedModelIDs) {
        glm::vec3 worldMin(0.0f);
        glm::vec3 worldMax(0.0f);
        if (modelRegistry.tryGetWorldBounds(id, worldMin, worldMax)) {
            gizmoPosition += (worldMin + worldMax) * 0.5f;
            count++;
        }
    }

    if (count > 0) {
        gizmoPosition /= static_cast<float>(count);
        return gizmoPosition;
    }

    for (uint32_t modelId : modelRegistry.getRenderableModelIds()) {
        glm::vec3 worldMin(0.0f);
        glm::vec3 worldMax(0.0f);
        if (modelRegistry.tryGetWorldBounds(modelId, worldMin, worldMax)) {
            return (worldMin + worldMax) * 0.5f;
        }
    }

    return glm::vec3(0.0f);
}
