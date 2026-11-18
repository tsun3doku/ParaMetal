#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <optional>

#include "interaction/IViewportInteraction.hpp"

class Camera;
struct WindowRuntimeState;

struct DomainFaceSelection {
    uint32_t nodeId = 0;
    int face = -1;
};

class DomainFaceInteractionController : public IViewportInteraction {
public:
    DomainFaceInteractionController(Camera& camera, const WindowRuntimeState& windowState);
    ~DomainFaceInteractionController() = default;

    void setEditingNode(int nodeId, bool armed);
    void setDomainBox(const std::array<glm::vec3, 8>& boxCorners, bool valid);

    bool handlePointerPress(float mouseX, float mouseY, int button = 1, bool shiftPressed = false) override;
    bool handlePointerMove(float mouseX, float mouseY) override;
    bool handlePointerRelease(float mouseX, float mouseY, int button = 1) override;
    bool hasPointerCapture() const override { return false; }

    int editingNodeId() const { return editingNode; }
    int selectedFace() const { return activeFace; }
    int hoverFace() const { return hoveredFace; }
    const std::array<glm::vec3, 8>& domainCorners() const { return corners; }

    std::optional<DomainFaceSelection> takePendingSelection();

private:
    int hitFace(float mouseX, float mouseY) const;

    Camera& camera;
    const WindowRuntimeState& windowState;

    int editingNode = 0;
    int activeFace = -1;
    int hoveredFace = -1;
    bool pickingArmed = false;
    bool boxValid = false;
    std::array<glm::vec3, 8> corners{};
    std::optional<DomainFaceSelection> pendingSelection;
};
