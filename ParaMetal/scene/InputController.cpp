#include "InputController.hpp"

#include "Camera.hpp"
#include "CameraController.hpp"
#include "interaction/InteractionContext.hpp"
#include "ModelSelection.hpp"
#include "render/WindowRuntimeState.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <algorithm>

InputController::InputController(
    CameraController& cameraController,
    InteractionContext& interactionContext,
    ModelSelection& modelSelection,
    ModelRegistry& modelRegistry,
    const WindowRuntimeState& windowState)
    : cameraController(cameraController),
      camera(cameraController.getCamera()),
      interactionContext(interactionContext),
      modelSelection(modelSelection),
      modelRegistry(modelRegistry),
      windowState(windowState) {
}

std::vector<ViewportCommand> InputController::takePendingViewportCommands() {
    std::vector<ViewportCommand> commands;
    commands.swap(pendingViewportCommands);
    return commands;
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

    if (key == Qt::Key_Escape) {
        interactionContext.cancelActiveInteractions();
        return;
    }

    if (key == Qt::Key_H) {
        pendingViewportCommands.push_back(ViewportCommand::ToggleWireframe);
    }
    else if (key == Qt::Key_AsciiTilde) {
        pendingViewportCommands.push_back(ViewportCommand::ToggleTimingOverlay);
    }
    else if (key == Qt::Key_F) {
        if (modelSelection.getSelected()) {
            const uint32_t selectedID = modelSelection.getSelectedModelID();
            glm::vec3 worldMin(0.0f);
            glm::vec3 worldMax(0.0f);
            if (modelRegistry.tryGetWorldBounds(selectedID, worldMin, worldMax)) {
                camera.setLookAt((worldMin + worldMax) * 0.5f);
            }
        }
    }
    else if (key == Qt::Key_G) {
        if (ctrlPressed) {
            cameraController.focusWorldOrigin();
        } else {
            pendingViewportCommands.push_back(ViewportCommand::ToggleGrid);
        }
    }
}

void InputController::handleMouseMove(float mouseX, float mouseY) {
    interactionContext.handlePointerMove(mouseX, mouseY);
}

void InputController::handleMouseRelease(int button, float mouseX, float mouseY) {
    const int interactionButton = (button == static_cast<int>(Qt::LeftButton)) ? 1 :
                                 (button == static_cast<int>(Qt::RightButton)) ? 2 : button;

    interactionContext.handlePointerRelease(mouseX, mouseY, interactionButton);
}

void InputController::handleMouseButton(int button, float mouseX, float mouseY, bool shiftPressed) {
    const int interactionButton = (button == static_cast<int>(Qt::LeftButton)) ? 1 :
                                 (button == static_cast<int>(Qt::RightButton)) ? 2 : button;

    if (interactionContext.handlePointerPress(mouseX, mouseY, interactionButton, shiftPressed)) {
        return;
    }

    if (button == static_cast<int>(Qt::LeftButton)) {
        interactionContext.clearSelections();
    }

    if (button != static_cast<int>(Qt::LeftButton)) {
        return;
    }

    const VkExtent2D swapChainExtent = viewportExtent(windowState);
    int x = static_cast<int>(mouseX);
    int y = static_cast<int>(mouseY);

    x = (std::max)(0, (std::min)(x, static_cast<int>(swapChainExtent.width) - 1));
    y = (std::max)(0, (std::min)(y, static_cast<int>(swapChainExtent.height) - 1));

    modelSelection.queuePickRequest(x, y, shiftPressed, mouseX, mouseY);
}

void InputController::processRawDrag(bool panActive, float dx, float dy) {
    cameraController.cancelTransition();
    if (dx != 0.0f || dy != 0.0f) {
        camera.applyDragDelta(dx, dy, panActive);
    }
}
