#pragma once

#include <vulkan/vulkan.h>
#include <memory>
#include <vector>

class CameraController;
class ModelRegistry;
struct WindowRuntimeState;
class NavigationGizmoController;
class MarkerInteractionController;
class SubselectionInteractionController;
class SubselectionPickService;
class DomainFaceInteractionController;
class GizmoController;
class IViewportInteraction;

class InteractionContext {
public:
    InteractionContext();
    ~InteractionContext();

    bool initialize(
        CameraController& cameraController,
        ModelRegistry& modelRegistry,
        const WindowRuntimeState& windowState);
    void shutdown();
    bool isInitialized() const { return initialized; }

    NavigationGizmoController& navigationGizmoController();
    const NavigationGizmoController& navigationGizmoController() const;

    MarkerInteractionController& markerInteractionController();
    const MarkerInteractionController& markerInteractionController() const;

    SubselectionInteractionController& subselectionInteractionController();
    const SubselectionInteractionController& subselectionInteractionController() const;

    void setSubselectionPickService(SubselectionPickService* pickService);

    DomainFaceInteractionController& domainFaceInteractionController();
    const DomainFaceInteractionController& domainFaceInteractionController() const;

    GizmoController& gizmoController();
    const GizmoController& gizmoController() const;

    bool handlePointerPress(float mouseX, float mouseY, int button = 1, bool shiftPressed = false);
    bool handlePointerMove(float mouseX, float mouseY);
    bool handlePointerRelease(float mouseX, float mouseY, int button = 1);

    bool hasActivePointerCapture() const;
    void cancelActiveInteractions();
    void clearSelections();

    void setViewport(VkExtent2D extent, float dpiScale);
    void resolvePick(const class ModelSelection& modelSelection, ModelRegistry& modelRegistry);

    void startProbePicking(uint32_t targetNodeId, int probeIndex = 0);
    void cancelProbePicking();
    bool isProbePickingActive() const;

    void setBoundaryFaceEditing(int nodeId, bool armed);

private:
    std::unique_ptr<NavigationGizmoController> navigationGizmo;
    std::unique_ptr<MarkerInteractionController> markerInteraction;
    std::unique_ptr<SubselectionInteractionController> subselection;
    std::unique_ptr<DomainFaceInteractionController> domainFace;
    std::unique_ptr<GizmoController> gizmo;

    std::vector<IViewportInteraction*> interactionStack;
    bool initialized = false;
};
