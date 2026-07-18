#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "app/AppTypes.hpp"
#include "RenderSettingsController.hpp"
#include "RenderSettingsManager.hpp"
#include "RenderContext.hpp"
#include "RuntimeController.hpp"
#include "RuntimeInterfaces.hpp"
#include "SimulationError.hpp"
#include "SceneContext.hpp"
#include "TimelineController.hpp"
#include "TimelineRuntime.hpp"
#include "VulkanCoreContext.hpp"

class NodeGraph;
class CameraController;
class ModelSelection;
class NodeGraphController;
class RuntimeQuery;
class SceneController;
struct WindowRuntimeState;

class RuntimeSystems : public RuntimeQuery {
public:
    RuntimeSystems() = default;
    ~RuntimeSystems();

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
    NodeGraphController* getNodeGraphController();
    const NodeGraphController* getNodeGraphController() const;

private:
    void cleanup();
    bool isSimulationActive() const override;
    bool isSimulationPaused() const override;
    float getSimulationTotalTime() const override;
    uint32_t getSimulationRecordedTimelineFrames() const override;
    uint32_t getSimulationTimelineFrameCount() const override;
    float getSimulationDuration() const override;
    uint32_t getSimulationResetCounter() const override;
    uint32_t getSimulationRewindFrame() const override;
    bool isTimelinePlaying() const override;
    uint32_t getTimelineCurrentFrame() const override;
    uint32_t getTimelineFrameCount() const override;
    uint32_t getTimelineStartDisplayFrame() const override;
    uint32_t getTimelineEndDisplayFrame() const override;
    float getTimelineCurrentSeconds() const override;
    float getTimelineDurationSeconds() const override;
    bool getSerialTemperatureStatus(
        uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const override;

    WindowRuntimeState* windowRuntimeState = nullptr;
    bool initialized = false;
    std::atomic<bool> simPaused{false};
    std::atomic<bool> runtimeBusy{false};
    std::atomic<bool> isShuttingDown{false};
    uint32_t frameCounter = 0;

    VulkanCoreContext core;
    SceneContext scene;
    RenderContext render;
    RuntimeController runtimeController;

    RenderSettingsManager settingsManager;
    RenderSettingsController settingsController{&settingsManager};
    TimelineRuntime timelineRuntime;
    TimelineController timelineControllerInstance{&timelineRuntime};
};
