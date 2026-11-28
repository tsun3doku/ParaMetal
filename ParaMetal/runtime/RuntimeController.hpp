#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>


#include "RuntimeEvent.hpp"
#include "RuntimeRenderController.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include "scene/InputController.hpp"

class ComputeContext;
class InteractionContext;
class NodeGraphContext;
class RawMouseInput;
class RenderContext;
class SceneContext;
class SceneController;
class SubselectionInteractionController;
class ViewportEditController;
class VulkanCoreContext;
struct SimTickContext;
struct WindowRuntimeState;

class RuntimeController {
public:
    RuntimeController();
    ~RuntimeController();

    bool initialize(RenderContext& render, ComputeContext& compute, NodeGraphContext& nodeGraph,
                    SceneContext& scene, InteractionContext& interaction, VulkanCoreContext& core,
                    WindowRuntimeState& windowRuntimeState, RawMouseInput* rawMouseInput);
    void shutdown();
    bool isInitialized() const;
    void tick(
        float deltaTime,
        uint32_t& frameCounter,
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const app::RenderSettings& renderSettings,
        const SimTickContext& simContext);

    std::vector<ViewportCommand> takePendingViewportCommands();
    void setSelectedNodes(
        const std::vector<NodeGraphNodeId>& nodeIds,
        int primaryNodeId);
    void invalidateSelection();
    void reconcileSelection();
    bool beginNodeAction(
        NodeGraphNodeId nodeId,
        uint64_t nodeInstanceId,
        uint32_t actionId);
    bool beginViewportSelection();
    bool selectionCommitRequired() const;
    void completeNodeAction();
    void cancelNodeAction();
    void clearNodeActionSelection();
    void setSubselectionElementType(int elementType);
    bool takePendingNodeParameters(
        NodeGraphNodeId& outNodeId,
        uint64_t& outOriginInstanceId,
        std::vector<NodeGraphParamValue>& outParameters);
    std::vector<RuntimeEvent> takePendingRuntimeEvents();

private:
    bool initializeInputPipeline(WindowRuntimeState& windowRuntimeState);
    void dispatchInputEvents();
    void updateRawMouseDrag();
    void collectRuntimeEvents();
    void simulateFrame(bool allowSimulation, const SimTickContext& simContext);
    void syncSubselectionDisplay(
        const SubselectionInteractionController& subselection);

    std::unique_ptr<ViewportEditController> viewportEdits;
    std::unique_ptr<InputController> inputControllerState;
    std::unique_ptr<SceneController> sceneControllerState;
    std::unique_ptr<RuntimeRenderController> runtimeRenderController;
    RenderContext* render = nullptr;
    ComputeContext* compute = nullptr;
    NodeGraphContext* nodeGraph = nullptr;
    SceneContext* scene = nullptr;
    InteractionContext* interaction = nullptr;
    WindowRuntimeState* windowRuntimeState = nullptr;
    RawMouseInput* rawMouseInput = nullptr;
    bool rawDragActive = false;
    bool initialized = false;
    int lastInspectedNodeId = 0;
    uint64_t lastInspectionGraphRevision = UINT64_MAX;
    uint64_t lastInspectionEvaluationRevision = UINT64_MAX;
    uint64_t lastInspectionModelRevision = UINT64_MAX;
    std::vector<RuntimeEvent> pendingRuntimeEvents;
    uint64_t appliedIntrinsicSelectionSocketKey = 0;
};
