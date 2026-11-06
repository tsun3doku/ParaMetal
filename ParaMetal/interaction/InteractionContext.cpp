#include "InteractionContext.hpp"

#include "IViewportInteraction.hpp"
#include "scene/CameraController.hpp"
#include "scene/GizmoController.hpp"
#include "scene/MarkerInteractionController.hpp"
#include "scene/NavigationGizmoController.hpp"
#include "vulkan/ModelRegistry.hpp"

InteractionContext::InteractionContext() = default;
InteractionContext::~InteractionContext() = default;

bool InteractionContext::initialize(
    CameraController& cameraController,
    ModelRegistry& modelRegistry,
    const WindowRuntimeState& windowState) {
    if (initialized) {
        return true;
    }

    navigationGizmo = std::make_unique<NavigationGizmoController>(cameraController);
    markerInteraction = std::make_unique<MarkerInteractionController>(
        cameraController.getCamera(), modelRegistry, windowState);
    gizmo = std::make_unique<GizmoController>(cameraController, windowState);

    // Interaction priority order:
    // 1. Screen-space widgets (Navigation cube)
    // 2. Entity manipulators (Probe marker placement and direct dragging)
    // 3. Transform gizmos (Translation/Rotation handles)
    interactionStack = {
        navigationGizmo.get(),
        markerInteraction.get(),
        gizmo.get()
    };

    initialized = true;
    return true;
}

void InteractionContext::shutdown() {
    interactionStack.clear();
    gizmo.reset();
    markerInteraction.reset();
    navigationGizmo.reset();
    initialized = false;
}

NavigationGizmoController& InteractionContext::navigationGizmoController() {
    return *navigationGizmo;
}

const NavigationGizmoController& InteractionContext::navigationGizmoController() const {
    return *navigationGizmo;
}

MarkerInteractionController& InteractionContext::markerInteractionController() {
    return *markerInteraction;
}

const MarkerInteractionController& InteractionContext::markerInteractionController() const {
    return *markerInteraction;
}

GizmoController& InteractionContext::gizmoController() {
    return *gizmo;
}

const GizmoController& InteractionContext::gizmoController() const {
    return *gizmo;
}

bool InteractionContext::handlePointerPress(float mouseX, float mouseY, int button, bool shiftPressed) {
    for (auto* tool : interactionStack) {
        if (tool && tool->hasPointerCapture()) {
            return tool->handlePointerPress(mouseX, mouseY, button, shiftPressed);
        }
    }
    for (auto* tool : interactionStack) {
        if (tool && tool->handlePointerPress(mouseX, mouseY, button, shiftPressed)) {
            return true;
        }
    }
    return false;
}

bool InteractionContext::handlePointerMove(float mouseX, float mouseY) {
    for (auto* tool : interactionStack) {
        if (tool && tool->hasPointerCapture()) {
            return tool->handlePointerMove(mouseX, mouseY);
        }
    }
    for (auto* tool : interactionStack) {
        if (tool && tool->handlePointerMove(mouseX, mouseY)) {
            return true;
        }
    }
    return false;
}

bool InteractionContext::handlePointerRelease(float mouseX, float mouseY, int button) {
    for (auto* tool : interactionStack) {
        if (tool && tool->hasPointerCapture()) {
            return tool->handlePointerRelease(mouseX, mouseY, button);
        }
    }
    for (auto* tool : interactionStack) {
        if (tool && tool->handlePointerRelease(mouseX, mouseY, button)) {
            return true;
        }
    }
    return false;
}

bool InteractionContext::hasActivePointerCapture() const {
    for (const auto* tool : interactionStack) {
        if (tool && tool->hasPointerCapture()) {
            return true;
        }
    }
    return false;
}

void InteractionContext::cancelActiveInteractions() {
    if (markerInteraction && markerInteraction->isProbePickingActive()) {
        markerInteraction->cancelProbePicking();
    }
    if (gizmo) {
        gizmo->cancelDrag();
    }
}

void InteractionContext::clearSelections() {
    if (markerInteraction) {
        markerInteraction->clearSelection();
    }
}

void InteractionContext::setViewport(VkExtent2D extent, float dpiScale) {
    if (navigationGizmo) {
        navigationGizmo->setViewport(extent, dpiScale);
    }
}

void InteractionContext::resolvePick(const ModelSelection& modelSelection, ModelRegistry& modelRegistry) {
    if (gizmo) {
        gizmo->updateGizmo(modelSelection, modelRegistry);
    }
}

void InteractionContext::startProbePicking(uint32_t targetNodeId, int probeIndex) {
    if (markerInteraction) {
        markerInteraction->startProbePicking(targetNodeId, probeIndex);
    }
}

void InteractionContext::cancelProbePicking() {
    if (markerInteraction) {
        markerInteraction->cancelProbePicking();
    }
}

bool InteractionContext::isProbePickingActive() const {
    return markerInteraction ? markerInteraction->isProbePickingActive() : false;
}
