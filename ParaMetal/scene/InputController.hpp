#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <Qt>
#include <cstdint>
#include <vector>

class Camera;
class CameraController;
class InteractionContext;
class ModelSelection;
class ModelRegistry;
struct WindowRuntimeState;

enum class ViewportCommand : uint8_t {
    ToggleWireframe,
    ToggleTimingOverlay,
    ToggleGrid
};

class InputController {
public:
    InputController(
        CameraController& cameraController,
        InteractionContext& interactionContext,
        ModelSelection& modelSelection,
        ModelRegistry& modelRegistry,
        const WindowRuntimeState& windowState);
    ~InputController() = default;

    void handleScrollInput(double yOffset);
    void handleKeyInput(Qt::Key key, bool pressed, bool ctrlPressed);
    void handleMouseMove(float mouseX, float mouseY);
    void handleMouseRelease(int button, float mouseX, float mouseY);
    void handleMouseButton(int button, float mouseX, float mouseY, bool shiftPressed);

    void processRawDrag(bool panActive, float dx, float dy);
    std::vector<ViewportCommand> takePendingViewportCommands();

private:
    CameraController& cameraController;
    Camera& camera;
    InteractionContext& interactionContext;
    ModelSelection& modelSelection;
    ModelRegistry& modelRegistry;
    const WindowRuntimeState& windowState;

    std::vector<ViewportCommand> pendingViewportCommands;
};
