#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <Qt>
#include <cstdint>
#include <vector>

#include "InputActions.hpp"
#include "nodegraph/NodeGraphTypes.hpp"

class Camera;
class CameraController;
class GizmoController;
class NavigationGizmoController;
class NodeGraphController;
class ModelSelection;
class ModelRegistry;
class SceneController;
struct WindowRuntimeState;

class InputController {
public:
    InputController(CameraController& cameraController,
        GizmoController& gizmoController,
        NavigationGizmoController& navigationGizmoController,
        ModelSelection& modelSelection,
        ModelRegistry& resourceManager,
        SceneController& sceneController,
        NodeGraphController& graphController,
        const WindowRuntimeState& windowState);
    ~InputController() = default;

    void handleScrollInput(double yOffset);
    void handleKeyInput(Qt::Key key, bool pressed, bool ctrlPressed);
    void handleMouseMove(float mouseX, float mouseY);
    void handleMouseRelease(int button, float mouseX, float mouseY);
    void handleMouseButton(int button, float mouseX, float mouseY, bool shiftPressed);

    void processInput(bool shiftPressed, bool middleButtonPressed, double mouseX, double mouseY, float deltaTime);
    void updateGizmo();
    std::vector<InputAction> takePendingActions();

    bool resolveSelectedTransformNode(NodeGraphNodeId& outTransformNodeId);

    bool isDraggingGizmo = false;

private:
    CameraController& cameraController;
    Camera& camera;
    GizmoController& gizmoController;
    NavigationGizmoController& navigationGizmoController;
    ModelSelection& modelSelection;
    ModelRegistry& resourceManager;
    SceneController& sceneController;
    NodeGraphController& graphController;
    const WindowRuntimeState& windowState;
    std::vector<InputAction> pendingActions;

    glm::vec3 accumulatedTranslation{0.0f};
    glm::vec3 lastAppliedTranslation{0.0f};
    float accumulatedRotation = 0.0f;
    float lastAppliedRotation = 0.0f;
    glm::vec3 cachedGizmoPosition{0.0f};
    NodeGraphNodeId activeTransformNodeId{};
    glm::vec3 transformDragStartTranslation{0.0f};
    glm::vec3 transformDragStartRotationDegrees{0.0f};
};


