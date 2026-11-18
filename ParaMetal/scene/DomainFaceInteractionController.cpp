#include "DomainFaceInteractionController.hpp"

#include "Camera.hpp"
#include "MousePicker.hpp"
#include "render/WindowRuntimeState.hpp"

DomainFaceInteractionController::DomainFaceInteractionController(
    Camera& camera,
    const WindowRuntimeState& windowState)
    : camera(camera),
      windowState(windowState) {
}

void DomainFaceInteractionController::setEditingNode(int nodeId, bool armed) {
    if (nodeId <= 0) {
        editingNode = 0;
        activeFace = -1;
        hoveredFace = -1;
        pickingArmed = false;
        return;
    }

    editingNode = nodeId;
    hoveredFace = -1;
    pickingArmed = armed;
}

void DomainFaceInteractionController::setDomainBox(const std::array<glm::vec3, 8>& boxCorners, bool valid) {
    corners = boxCorners;
    boxValid = valid;
    if (!valid) {
        hoveredFace = -1;
    }
}

int DomainFaceInteractionController::hitFace(float mouseX, float mouseY) const {
    if (!boxValid || mouseX < 0.0f || mouseY < 0.0f) {
        return -1;
    }

    const uint32_t width = windowState.width.load(std::memory_order_acquire);
    const uint32_t height = windowState.height.load(std::memory_order_acquire);
    if (width == 0 || height == 0) {
        return -1;
    }

    // Corner 0 is the minimum corner, corner 7 the maximum
    const Ray ray = MousePicker::screenToWorldRay(mouseX, mouseY, width, height, camera);
    float t = 0.0f;
    int face = -1;
    if (!MousePicker::rayIntersectsAABBFace(ray, corners[0], corners[7], t, face)) {
        return -1;
    }
    return face;
}

bool DomainFaceInteractionController::handlePointerPress(float mouseX, float mouseY, int button, bool shiftPressed) {
    (void)shiftPressed;
    if (editingNode == 0 || !pickingArmed) {
        return false;
    }

    if (button == 2) {
        if (activeFace < 0 && hoveredFace < 0) {
            return false;
        }
        activeFace = -1;
        hoveredFace = -1;
        pendingSelection = DomainFaceSelection{ static_cast<uint32_t>(editingNode), -1 };
        return true;
    }

    if (button != 1) {
        return false;
    }

    const int face = hitFace(mouseX, mouseY);
    if (face < 0) {
        return false;
    }

    hoveredFace = face;
    if (face != activeFace) {
        activeFace = face;
        pendingSelection = DomainFaceSelection{ static_cast<uint32_t>(editingNode), face };
    }
    return true;
}

bool DomainFaceInteractionController::handlePointerMove(float mouseX, float mouseY) {
    if (editingNode == 0 || !pickingArmed || mouseX < 0.0f || mouseY < 0.0f) {
        hoveredFace = -1;
        return false;
    }

    hoveredFace = hitFace(mouseX, mouseY);
    return false;
}

bool DomainFaceInteractionController::handlePointerRelease(float mouseX, float mouseY, int button) {
    (void)mouseX;
    (void)mouseY;
    (void)button;
    return false;
}

std::optional<DomainFaceSelection> DomainFaceInteractionController::takePendingSelection() {
    if (!pendingSelection.has_value()) {
        return std::nullopt;
    }
    const DomainFaceSelection selection = *pendingSelection;
    pendingSelection.reset();
    return selection;
}
