#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>

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
    bool takePendingNodeParameters(
        NodeGraphNodeId& outNodeId,
        std::vector<NodeGraphParamValue>& outParameters);

private:
    bool initializeInputPipeline(WindowRuntimeState& windowRuntimeState);
    void dispatchInputEvents();
    void updateRawMouseDrag();
    void simulateFrame(bool allowSimulation, const SimTickContext& simContext);

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
};
