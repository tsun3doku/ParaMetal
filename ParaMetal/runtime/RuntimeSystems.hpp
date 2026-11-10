#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "app/AppTypes.hpp"
#include "app/RawMouseInput.hpp"
#include "ComputeContext.hpp"
#include "framegraph/FrameSync.hpp"
#include "NodeGraphContext.hpp"
#include "RenderContext.hpp"
#include "RuntimeController.hpp"
#include "runtime/export/RuntimeExport.hpp"
#include "RuntimeInterfaces.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include "SceneContext.hpp"
#include "interaction/InteractionContext.hpp"
#include "TimelineController.hpp"
#include "TimelineRuntime.hpp"
#include "VulkanCoreContext.hpp"
#include "util/Units.hpp"

class CameraController;
enum class CameraProjectionMode : uint8_t;
class ModelSelection;
class NodeGraphController;
struct WindowRuntimeState;

class RuntimeSystems : public RuntimeQuery {
public:
    RuntimeSystems();
    ~RuntimeSystems();

    bool initialize(WindowRuntimeState& runtimeState, const AppVulkanContext& vulkanContext);
    void tickFrame(float deltaTime, VkCommandBuffer commandBuffer, uint32_t frameIndex);
    bool updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent);
    bool replaceGraphState(const NodeGraphState& state);
    bool applyGraphDelta(const NodeGraphDelta& delta);
    bool takePendingNodeParameters(NodeGraphNodeId& outNodeId, std::vector<NodeGraphParamValue>& outParameters);
    void requestExport(int nodeId);
    void requestExportProducts(int nodeId, uint64_t requestId, uint64_t graphRevision);
    void processDeferredWork();
    bool takePendingExportResult(int& outNodeId, bool& outSuccess, std::string& outMessage);
    bool takeExportProductsReady(ExportProductsReply& outReply);
    void shutdown();
    bool isInitialized() const;

    const RuntimeQuery* runtimeQuery() const;
    TimelineController* timelineController();
    void setWireframeMode(app::WireframeMode mode);
    void setGridEnabled(bool enabled);
    void setBackgroundMode(app::BackgroundMode mode);
    void setNavigationCubeVisible(bool visible);
    void setAxisLabelsVisible(bool visible);
    void focusCameraOnWorldOrigin();
    void setCameraProjectionMode(CameraProjectionMode mode);
    void setCameraFov(float degrees);
    void setCameraZoomSpeed(float speed);
    void setCameraPanSpeed(float speed);
    void setHeatPaletteRange(float minimum, float maximum);
    void setHeatPalette(int palette);
    void setVelocityPaletteRange(float minimum, float maximum);
    void setVelocityPalette(int palette);
    void setWorldUnit(int unit);
    bool isHeatPaletteVisible() const;
    bool isVelocityPaletteVisible() const;
    const app::RenderSettings& renderSettings() const;
    CameraController* getCameraController();
    ModelSelection* getModelSelection();
    NodeGraphController* getNodeGraphController();
    InteractionContext& getInteractionContext() { return interaction; }
    void noteFramePresented();
    void startProbePicking(uint32_t targetNodeId, int probeIndex = 0);
    void cancelProbePicking();

private:
    void cleanup();
    void dispatchViewportCommands();
    void reclaimRetiredGenerations();
    bool isSimulationActive() const override;
    bool isSimulationPaused() const override;
    uint32_t getSimulationRecordedTimelineFrames() const override;
    bool isTimelinePlaying() const override;
    uint32_t getTimelineCurrentFrame() const override;
    uint32_t getTimelineFrameCount() const override;
    uint32_t getTimelineStartDisplayFrame() const override;
    uint32_t getTimelineEndDisplayFrame() const override;
    float getTimelineCurrentSeconds() const override;
    float getTimelineDurationSeconds() const override;
    bool getSerialTemperatureStatus(uint64_t sourceKey, SerialTemperatureRuntime::Status& outStatus) const override;
    bool getHeatProbeTemperatures(uint64_t sourceKey, std::vector<float>& outTemperaturesC) const override;

    WindowRuntimeState* windowRuntimeState = nullptr;
    bool initialized = false;
    std::atomic<bool> runtimeBusy{false};
    std::atomic<bool> isShuttingDown{false};
    uint32_t frameCounter = 0;

    VulkanCoreContext core;
    SceneContext scene;
    InteractionContext interaction;
    FrameSync frameSync;
    RawMouseInput rawMouseInput;
    ComputeContext compute;
    RenderContext render;
    NodeGraphContext nodeGraph;
    RuntimeController runtimeController;

    app::RenderSettings renderSettingsState;
    units::LengthUnit worldUnitState = units::defaultLengthUnit();
    TimelineRuntime timelineRuntime;
    TimelineController timelineControllerInstance{&timelineRuntime};
    bool hadSimulationTimeline = false;
    RuntimeExport runtimeExport;
};
