#pragma once

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <cstdint>
#include <optional>
#include "interaction/IViewportInteraction.hpp"

class CameraController;
class Camera;
class ModelSelection;
class ModelRegistry;
struct WindowRuntimeState;

enum class GizmoAxis {
    None = 0,
    X = 1,
    Y = 2,
    Z = 3
};

enum class GizmoMode {
    Translate,
    Rotate,
    Scale
};

struct GizmoDragBegin {
    uint32_t runtimeModelId = 0;
    GizmoMode mode = GizmoMode::Translate;
    GizmoAxis axis = GizmoAxis::None;
};

struct GizmoDragUpdate {
    glm::vec3 translationDeltaWorld{0.0f};
    float rotationDeltaDegrees = 0.0f;
};

class GizmoController : public IViewportInteraction {
public:
    GizmoController(
        const CameraController& cameraController,
        const WindowRuntimeState& windowState);
    ~GizmoController() = default;

    void setMode(GizmoMode mode) {
        currentMode = mode;
    }

    GizmoMode getMode() const {
        return currentMode;
    }

    void setActiveAxis(GizmoAxis axis) {
        activeAxis = axis;
    }

    GizmoAxis getActiveAxis() const {
        return activeAxis;
    }

    bool isActive() const {
        return activeAxis != GizmoAxis::None;
    }

    bool handlePointerPress(float mouseX, float mouseY, int button = 1, bool shiftPressed = false) override;
    bool handlePointerMove(float mouseX, float mouseY) override;
    bool handlePointerRelease(float mouseX, float mouseY, int button = 1) override;
    bool hasPointerCapture() const override { return isDraggingGizmo; }

    void updateGizmo(const ModelSelection& modelSelection, ModelRegistry& modelRegistry);

    std::optional<GizmoDragBegin> takePendingDragBegin();
    std::optional<GizmoDragUpdate> takePendingDragUpdate();
    bool takePendingDragEnd();
    void cancelDrag();

    void startDrag(GizmoAxis axis, const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition);
    void endDrag();

    glm::vec3 calculateTranslationDelta(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition, GizmoAxis axis);
    float calculateRotationDelta(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& gizmoPosition, GizmoAxis axis);

    glm::vec3 calculateGizmoPosition(ModelRegistry& modelRegistry, const ModelSelection& modelSelection);

private:
    const CameraController& cameraController;
    const Camera& camera;
    const WindowRuntimeState& windowState;

    GizmoMode currentMode = GizmoMode::Translate;
    GizmoAxis activeAxis = GizmoAxis::None;

    bool isDraggingGizmo = false;
    glm::vec3 cachedGizmoPosition{0.0f};
    glm::vec3 accumulatedTranslation{0.0f};
    float accumulatedRotation = 0.0f;

    std::optional<GizmoDragBegin> pendingGizmoDragBegin;
    std::optional<GizmoDragUpdate> pendingGizmoDragUpdate;
    bool pendingGizmoDragEnd = false;

    glm::vec3 dragStartPos = glm::vec3(0.0f);
    glm::vec3 dragStartRayOrigin = glm::vec3(0.0f);
    glm::vec3 dragStartRayDir = glm::vec3(0.0f);
    glm::vec3 dragStartIntersection = glm::vec3(0.0f);
};
