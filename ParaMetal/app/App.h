#pragma once

#include <memory>
#include <string>
#include <vector>

#include "app/AppTypes.hpp"
#include "runtime/SimulationError.hpp"
#include "runtime/RuntimeSystems.hpp"

class NodeGraph;
class CameraController;
class ModelSelection;
class RenderSettingsController;
class RuntimeQuery;
class SceneController;
class TimelineController;
struct WindowRuntimeState;

class App {
public:
    App();
    ~App();

    bool initialize(WindowRuntimeState& runtimeState, const AppVulkanContext& vulkanContext);
    void tickFrame(float deltaTime);
    void shutdown();
    bool isInitialized() const;

    const RuntimeQuery* runtimeQuery() const;
    TimelineController* timelineController();
    const TimelineController* timelineController() const;
    std::vector<SimulationError> consumeSimulationErrors();
    uint32_t loadModel(const std::string& modelPath, uint32_t preferredModelId = 0);
    void setPanSensitivity(float sensitivity);
    void setSimPaused(bool paused);
    RenderSettingsController* getSettingsController();
    const RenderSettingsController* getSettingsController() const;
    NodeGraph* getNodeGraph();
    const NodeGraph* getNodeGraph() const;
    CameraController* getCameraController();
    const CameraController* getCameraController() const;
    SceneController* getSceneController();
    const SceneController* getSceneController() const;
    ModelSelection* getModelSelection();
    const ModelSelection* getModelSelection() const;

private:
    std::unique_ptr<RuntimeSystems> systems;
};
