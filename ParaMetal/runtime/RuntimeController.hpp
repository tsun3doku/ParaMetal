#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "RuntimeInput.hpp"
#include "RuntimeRenderController.hpp"
#include "framegraph/FrameController.hpp"
#include "scene/InputController.hpp"

class ComputeContext;
class NodeGraphContext;
class RawMouseInput;
class RenderContext;
class SceneContext;
class SceneController;
class VulkanCoreContext;
class NodeGraphController;
class CameraController;
struct WindowRuntimeState;

class RuntimeController {
public:
    RuntimeController();
    ~RuntimeController();

    bool initialize(RenderContext& render, ComputeContext& compute, NodeGraphContext& nodeGraph,
                    SceneContext& scene, VulkanCoreContext& core, WindowRuntimeState& windowRuntimeState,
                    RawMouseInput* rawMouseInput);
    void shutdown();
    bool isInitialized() const;
    void tick(
        float deltaTime,
        uint32_t& frameCounter,
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const app::RenderSettings& renderSettings);

    bool hasLastFrameSlot() const;
    uint32_t lastFrameSlot() const;
    std::vector<ViewportCommand> takePendingViewportCommands();
    bool takePendingNodeParameters(
        NodeGraphNodeId& outNodeId,
        std::vector<NodeGraphParamValue>& outParameters);

    SceneController* sceneController();
    const SceneController* sceneController() const;

private:
    bool initializeInputPipeline(WindowRuntimeState& windowRuntimeState);
    void simulateFrame(bool allowSimulation, uint32_t frameIndex);
    NodeGraphController* nodeGraphController();
    CameraController* cameraController();

    std::unique_ptr<RuntimeInput> runtimeInput;
    std::unique_ptr<InputController> inputControllerState;
    std::unique_ptr<SceneController> sceneControllerState;
    std::unique_ptr<RuntimeRenderController> runtimeRenderController;
    RenderContext* render = nullptr;
    ComputeContext* compute = nullptr;
    NodeGraphContext* nodeGraph = nullptr;
    SceneContext* scene = nullptr;
    bool hasFrameSlot = false;
    uint32_t frameSlot = 0;
    bool initialized = false;
};
