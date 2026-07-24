#include "InputController.hpp"

#include "Camera.hpp"
#include "CameraController.hpp"
#include "GizmoController.hpp"
#include "NavigationGizmoController.hpp"
#include "ModelSelection.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeTransformParams.hpp"
#include "render/WindowRuntimeState.hpp"
#include "scene/SceneController.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <algorithm>
#include <cmath>

bool InputController::resolveSelectedTransformNode(NodeGraphNodeId& outTransformNodeId) {
    outTransformNodeId = {};
    const auto& selectedIDs = modelSelection.getSelectedModelIDsRenderThread();
    if (selectedIDs.size() != 1) {
        return false;
    }

    uint64_t outputSocketKey = 0;
    if (!sceneController.tryGetRuntimeModelSocketKey(selectedIDs.front(), outputSocketKey) || outputSocketKey == 0) {
        return false;
    }

    return graphController.resolveGizmoTransformNode(outputSocketKey, outTransformNodeId);
}

InputController::InputController(CameraController& cameraController, GizmoController& gizmoController,
    NavigationGizmoController& navigationGizmoController, ModelSelection& modelSelection, ModelRegistry& resourceManager,
    SceneController& sceneController, NodeGraphController& graphController,
    const WindowRuntimeState& windowState)
    : cameraController(cameraController),
      camera(cameraController.getCamera()),
      gizmoController(gizmoController),
      navigationGizmoController(navigationGizmoController),
      modelSelection(modelSelection),
      resourceManager(resourceManager),
      sceneController(sceneController),
      graphController(graphController),
      windowState(windowState) {
}

std::vector<InputAction> InputController::takePendingActions() {
    std::vector<InputAction> actions;
    actions.swap(pendingActions);
    return actions;
}

static VkExtent2D viewportExtent(const WindowRuntimeState& windowState) {
    return {
        windowState.width.load(std::memory_order_acquire),
        windowState.height.load(std::memory_order_acquire)
    };
}

void InputController::handleScrollInput(double yOffset) {
    cameraController.cancelTransition();
    camera.processMouseScroll(yOffset);
}

void InputController::handleKeyInput(Qt::Key key, bool pressed, bool ctrlPressed) {
    if (!pressed) {
        return;
    }

    if (key == Qt::Key_H) {
        pendingActions.emplace_back(ToggleWireframeAction{});
    }
    else if (key == Qt::Key_AsciiTilde) {
        pendingActions.emplace_back(ToggleTimingOverlayAction{});
    }
    else if (key == Qt::Key_F) {
        if (modelSelection.getSelected()) {
            const uint32_t selectedID = modelSelection.getSelectedModelID();
            glm::vec3 worldCenter(0.0f);
            if (resourceManager.tryGetWorldBoundingBoxCenter(selectedID, worldCenter)) {
                camera.setLookAt(worldCenter);
            }
        }
    }
    else if (key == Qt::Key_G) {
        if (ctrlPressed) {
            camera.setLookAt(glm::vec3(0.0f));
            camera.resetRadius();
        } else {
            pendingActions.emplace_back(ToggleGridAction{});
        }
    }
}

void InputController::handleMouseMove(float mouseX, float mouseY) {
    if (navigationGizmoController.handlePointerMove(mouseX, mouseY)) {
        return;
    }
    const VkExtent2D swapChainExtent = viewportExtent(windowState);

    if (isDraggingGizmo) {
        const glm::vec3 rayOrigin = camera.screenToWorldRayOrigin(
            mouseX, mouseY, swapChainExtent.width, swapChainExtent.height);
        const glm::vec3 rayDir = camera.screenToWorldRay(mouseX, mouseY, swapChainExtent.width, swapChainExtent.height);

        if (gizmoController.getMode() == GizmoMode::Translate) {
            const glm::vec3 newTranslation = gizmoController.calculateTranslationDelta(rayOrigin, rayDir, cachedGizmoPosition, gizmoController.getActiveAxis());
            accumulatedTranslation = newTranslation;
        }
        else if (gizmoController.getMode() == GizmoMode::Rotate) {
            const float angle = gizmoController.calculateRotationDelta(rayOrigin, rayDir, cachedGizmoPosition, gizmoController.getActiveAxis());
            accumulatedRotation = angle;
        }
    }
}

void InputController::handleMouseRelease(int button, float mouseX, float mouseY) {
    if (button != static_cast<int>(Qt::LeftButton)) {
        return;
    }

    if (navigationGizmoController.handlePointerRelease(mouseX, mouseY)) {
        return;
    }

    if (isDraggingGizmo) {
        isDraggingGizmo = false;
        gizmoController.endDrag();
        accumulatedTranslation = glm::vec3(0.0f);
        lastAppliedTranslation = glm::vec3(0.0f);
        accumulatedRotation = 0.0f;
        lastAppliedRotation = 0.0f;
        activeTransformNodeId = {};
    }
}

void InputController::handleMouseButton(int button, float mouseX, float mouseY, bool shiftPressed) {
    if (button != static_cast<int>(Qt::LeftButton)) {
        return;
    }

    if (navigationGizmoController.handlePointerPress(mouseX, mouseY)) {
        return;
    }

    const VkExtent2D swapChainExtent = viewportExtent(windowState);
    int x = static_cast<int>(mouseX);
    int y = static_cast<int>(mouseY);

    x = (std::max)(0, (std::min)(x, static_cast<int>(swapChainExtent.width) - 1));
    y = (std::max)(0, (std::min)(y, static_cast<int>(swapChainExtent.height) - 1));

    modelSelection.queuePickRequest(x, y, shiftPressed, mouseX, mouseY);
}

void InputController::processInput(bool shiftPressed, bool middleButtonPressed, double mouseX, double mouseY, float deltaTime) {
    (void)deltaTime;
    if (middleButtonPressed) {
        cameraController.cancelTransition();
    }
    camera.processMouseMovement(middleButtonPressed, mouseX, mouseY, shiftPressed);
}

void InputController::updateGizmo() {
    const VkExtent2D swapChainExtent = viewportExtent(windowState);

    if (!isDraggingGizmo) {
        const PickedResult lastPick = modelSelection.getLastPickedResult();
        if (lastPick.isGizmo() && modelSelection.getSelected()) {
            GizmoAxis hitAxis = GizmoAxis::None;
            if (lastPick.gizmoAxis == PickedGizmoAxis::X) {
                hitAxis = GizmoAxis::X;
            }
            else if (lastPick.gizmoAxis == PickedGizmoAxis::Y) {
                hitAxis = GizmoAxis::Y;
            }
            else if (lastPick.gizmoAxis == PickedGizmoAxis::Z) {
                hitAxis = GizmoAxis::Z;
            }

            if (hitAxis != GizmoAxis::None) {
                NodeGraphNodeId transformNodeId{};
                if (!resolveSelectedTransformNode(transformNodeId)) {
                    modelSelection.clearLastPickedResult();
                    return;
                }

                const NodeGraphNode* transformNode = graphController.graphState().node(transformNodeId);
                if (!transformNode) {
                    modelSelection.clearLastPickedResult();
                    return;
                }
                const TransformNodeParams initialParams = readTransformNodeParams(*transformNode);
                glm::vec3 initialTranslation(
                    static_cast<float>(initialParams.translateX),
                    static_cast<float>(initialParams.translateY),
                    static_cast<float>(initialParams.translateZ));
                glm::vec3 initialRotationDegrees(
                    static_cast<float>(initialParams.rotateXDegrees),
                    static_cast<float>(initialParams.rotateYDegrees),
                    static_cast<float>(initialParams.rotateZDegrees));

                if (lastPick.gizmoMode == PickedGizmoMode::Translate) {
                    gizmoController.setMode(GizmoMode::Translate);
                }
                else if (lastPick.gizmoMode == PickedGizmoMode::Rotate) {
                    gizmoController.setMode(GizmoMode::Rotate);
                }

                const PickingRequest pickReq = modelSelection.getLastPickRequest();
                const glm::vec3 gizmoPosition = gizmoController.calculateGizmoPosition(resourceManager, modelSelection);
                const glm::vec3 rayOrigin = camera.screenToWorldRayOrigin(pickReq.mouseX, pickReq.mouseY, swapChainExtent.width, swapChainExtent.height);
                const glm::vec3 rayDir = camera.screenToWorldRay(pickReq.mouseX, pickReq.mouseY, swapChainExtent.width, swapChainExtent.height);

                isDraggingGizmo = true;
                cachedGizmoPosition = gizmoPosition;
                gizmoController.startDrag(hitAxis, rayOrigin, rayDir, cachedGizmoPosition);
                accumulatedTranslation = glm::vec3(0.0f);
                lastAppliedTranslation = glm::vec3(0.0f);
                accumulatedRotation = 0.0f;
                lastAppliedRotation = 0.0f;
                activeTransformNodeId = transformNodeId;
                transformDragStartTranslation = initialTranslation;
                transformDragStartRotationDegrees = initialRotationDegrees;

                modelSelection.clearLastPickedResult();
            }
        }
    }

    if (!isDraggingGizmo) {
        return;
    }
    if (!activeTransformNodeId.isValid()) {
        return;
    }

    if (gizmoController.getMode() == GizmoMode::Translate) {
        const glm::vec3 currentTranslation = accumulatedTranslation;
        if (glm::length(currentTranslation - lastAppliedTranslation) < 1e-6f) {
            return;
        }

        const glm::vec3 authoredTranslation = transformDragStartTranslation + currentTranslation;
        pendingActions.emplace_back(SetNodeParametersAction{
            activeTransformNodeId,
            {
                {nodegraphparams::transform::TranslateX, NodeGraphParamType::Float, authoredTranslation.x},
                {nodegraphparams::transform::TranslateY, NodeGraphParamType::Float, authoredTranslation.y},
                {nodegraphparams::transform::TranslateZ, NodeGraphParamType::Float, authoredTranslation.z}
            }});

        lastAppliedTranslation = currentTranslation;
    }
    else if (gizmoController.getMode() == GizmoMode::Rotate) {
        const float currentRotation = accumulatedRotation;
        const float deltaRotation = currentRotation - lastAppliedRotation;

        if (fabs(deltaRotation) < 0.01f) {
            return;
        }

        const GizmoAxis activeAxis = gizmoController.getActiveAxis();
        if (activeAxis != GizmoAxis::X &&
            activeAxis != GizmoAxis::Y &&
            activeAxis != GizmoAxis::Z) {
            return;
        }

        glm::vec3 authoredRotation = transformDragStartRotationDegrees;
        if (activeAxis == GizmoAxis::X) {
            authoredRotation.x += currentRotation;
        }
        else if (activeAxis == GizmoAxis::Y) {
            authoredRotation.y += currentRotation;
        }
        else if (activeAxis == GizmoAxis::Z) {
            authoredRotation.z += currentRotation;
        }

        pendingActions.emplace_back(SetNodeParametersAction{
            activeTransformNodeId,
            {
                {nodegraphparams::transform::RotateXDegrees, NodeGraphParamType::Float, authoredRotation.x},
                {nodegraphparams::transform::RotateYDegrees, NodeGraphParamType::Float, authoredRotation.y},
                {nodegraphparams::transform::RotateZDegrees, NodeGraphParamType::Float, authoredRotation.z}
            }});

        lastAppliedRotation = currentRotation;
    }
}

