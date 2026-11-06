#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <optional>
#include <vector>

#include "interaction/IViewportInteraction.hpp"

class Camera;
class ModelRegistry;
struct WindowRuntimeState;

struct MarkerItem {
    uint32_t nodeId = 0;
    int probeIndex = 0;
    uint32_t probeId = 0;
    glm::vec3 worldPosition{0.0f};
};

struct MarkerCommit {
    uint32_t nodeId = 0;
    int probeIndex = 0;
    glm::vec3 position{0.0f};
};

class MarkerInteractionController : public IViewportInteraction {
public:
    MarkerInteractionController(
        Camera& camera,
        ModelRegistry& modelRegistry,
        const WindowRuntimeState& windowState);
    ~MarkerInteractionController() = default;

    void updateMarkers(const std::vector<MarkerItem>& updatedMarkers);

    bool handlePointerPress(float mouseX, float mouseY, int button = 1, bool shiftPressed = false) override;
    bool handlePointerMove(float mouseX, float mouseY) override;
    bool handlePointerRelease(float mouseX, float mouseY, int button = 1) override;
    bool hasPointerCapture() const override { return isDragging; }
    glm::vec3 getLiveDragPosition() const { return dragCurrentPosition; }

    int getSelectedProbeIndex() const { return selectedProbeIndex; }
    uint32_t getSelectedNodeId() const { return selectedNodeId; }
    uint32_t getSelectedProbeId() const { return selectedProbeId; }
    void setSelected(uint32_t nodeId, int probeIndex, uint32_t probeId = 0);
    void clearSelection();

    void startProbePicking(uint32_t targetNodeId, int probeIndex = 0);
    void cancelProbePicking();
    bool isProbePickingActive() const { return probePickingActive; }

    std::optional<MarkerCommit> takePendingCommit();
    std::optional<uint32_t> takePendingSelectedNodeId();

    int hitTest(float mouseX, float mouseY) const;

private:
    Camera& camera;
    ModelRegistry& modelRegistry;
    const WindowRuntimeState& windowState;

    std::vector<MarkerItem> markers;

    bool isDragging = false;
    glm::vec3 dragCurrentPosition{0.0f};
    glm::vec3 dragStartPosition{0.0f};

    uint32_t selectedNodeId = 0;
    int selectedProbeIndex = -1;
    uint32_t selectedProbeId = 0;

    bool probePickingActive = false;
    uint32_t probePickingNodeId = 0;
    int probePickingIndex = 0;

    std::optional<MarkerCommit> pendingCommit;
    std::optional<uint32_t> pendingSelectedNodeId;
};
