#pragma once

#include "heat/TimelineSimulation.hpp"
#include "framegraph/ComputePass.hpp"
#include "simgraph/SimNode.hpp"
#include "util/Structs.hpp"
#include "util/Units.hpp"
#include "HeatPlaybackRuntime.hpp"
#include "HeatDomainRuntime.hpp"
#include "HeatFluidRuntime.hpp"
#include "heat/HeatFluidVolumeStage.hpp"
#include "voronoi/GlobalDomainTopology.hpp"

#include <chrono>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <array>
#include <glm/glm.hpp>

class MemoryAllocator;
class VulkanDevice;
class CommandPool;
class HeatSystemSurfaceStage;

class HeatSystem : public SimNode {
public:
    HeatSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator,
        uint32_t maxFramesInFlight, CommandPool& renderCommandPool, CommandPool& transferCommandPool);
    ~HeatSystem() override;

    void update() override;
    bool ensureConfigured();
    bool setupDescriptors(const std::vector<VkBuffer>& surfaceBuffers, const std::vector<VkDeviceSize>& surfaceOffsets, const std::vector<VkBuffer>& gradientBuffers, const std::vector<VkDeviceSize>& gradientOffsets);
    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;
    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);
    void cleanup();

    void setActive(bool active);
    void setRewindFrame(uint32_t frame);
    void setPlaybackState(bool paused, uint32_t resetCounter);
    void setParams(float simulationDuration);
    void setHeatModels(
        const std::vector<std::vector<glm::vec3>>& modelSurfacePositions,
        const std::vector<std::vector<glm::vec3>>& modelSurfaceNormals,
        const std::vector<std::vector<uint32_t>>& modelSurfaceTriangleIndices,
        const std::vector<uint32_t>& modelRuntimeModelIds,
        const std::unordered_map<uint32_t, float>& modelInitialTemperaturesCByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelVolumetricPowerDensitiesByRuntimeId,
        const std::unordered_map<uint32_t, bool>& modelPrescribedTemperatureEnabledByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelPrescribedTemperaturesCByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelDensity,
        const std::unordered_map<uint32_t, float>& modelSpecificHeat,
        const std::unordered_map<uint32_t, float>& modelConductivity,
        units::LengthUnit worldUnit);

    void clearGlobalVoronoiInput();

    void setHeatFluid(
        float density,
        float specificHeat,
        float conductivity,
        float initialTemperatureC);
    void clearHeatFluid();

    void setAdvectionInput(
        VkBuffer faceFluxBuffer,
        VkDeviceSize faceFluxBufferSize,
        VkBuffer boundaryFluxBuffer,
        VkDeviceSize boundaryFluxBufferSize,
        uint32_t faceCount,
        uint32_t boundaryFaceCount);
    void clearAdvectionInput();

    // SimNode consumer side for cross-simulation publication
    void bindConsumedInput(uint64_t connectionId, VkSemaphore producerReady, uint64_t producerReadyValue) override {
        (void)producerReady;
        (void)producerReadyValue;
        boundInputConnectionId = connectionId;
    }
    void clearConsumedInputs() override {
        boundInputConnectionId = 0;
    }
    bool consumedInput(uint64_t connectionId) const override {
        return connectionId != 0 &&
            connectionId == boundInputConnectionId &&
            domainRuntime.advectionStagingCopyScheduled();
    }

    void setGlobalVoronoiInput(
        const topology::GlobalDomain& domain,
        const std::unordered_map<uint32_t, std::array<float, 16>>& modelLocalToWorld);

    void resetSimulationState();
    void setDomainBoundaryConditions(
        const uint32_t types[6],
        const float temperaturesC[6],
        const float heatFluxes[6],
        const float heatTransferCoefficients[6]);
    void setDomainBoundaryConditions(const heat::DomainBoundaryConditions& boundaries);
    const heat::DomainBoundaryConditions& getDomainBoundaryConditions() const { return domainBoundaries; }
    void setGlobalSolverParameters(uint32_t maxIterations, float convergenceTolerance) {
        domainRuntime.setGlobalSolverParameters(maxIterations, convergenceTolerance);
    }
    void waitForSolvesIdle() { domainRuntime.waitGlobalSolveIdle(); }
    void releasePendingSolveWaitOnHost() { domainRuntime.releasePendingSolveWaitOnHost(); }
    void setInitialTemperatures(const std::unordered_map<uint32_t, float>& temperatures);
    bool setRuntimeFluidInitialTemperature(float temperatureC);
    bool setRuntimeVolumetricPowerDensity(uint32_t runtimeModelId, float powerDensity);
    bool setRuntimePrescribedTemperature(uint32_t runtimeModelId, float tempC);

    bool getIsActive() const { return isActive; }
    bool getIsPaused() const { return timeline.isPaused(); }
    bool isInitialized() const { return initialized; }
    bool hasDispatchableComputeWork() const override;
    float computeGpuMs() const override { return domainRuntime.getLastSolveMs(); }
    uint64_t completedStepCount() const override { return completedSteps; }
    bool advectionStagingCopyScheduled() const { return domainRuntime.advectionStagingCopyScheduled(); }
    bool voronoiReady() const;

    uint32_t getResetCounter() const { return timeline.getResetCounter(); }
    float getSimulationTimeSeconds() const { return timeline.getCurrentPosition(); }
    uint32_t getRecordedTimelineFrames() const;
    uint32_t getTimelineFrameCount() const;
    float getSimulationDurationSeconds() const { return timeline.getDuration(); }
    uint32_t getRewindFrame() const;

    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }
    ComputePass::Synchronization getSynchronization() const override;
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return domainRuntime.getActiveModels(); }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const { return domainRuntime.getModelByRuntimeId(runtimeModelId); }
    HeatFluidRuntime* getFluidRuntime() { return fluidRuntime.get(); }
    const HeatFluidRuntime* getFluidRuntime() const { return fluidRuntime.get(); }
    VkImageView getFluidTemperatureVolumeView() const {
        return fluidVolumeStage ? fluidVolumeStage->getTemperatureVolumeView() : VK_NULL_HANDLE;
    }
    VkSampler getFluidTemperatureVolumeSampler() const {
        return fluidVolumeStage ? fluidVolumeStage->getTemperatureVolumeSampler() : VK_NULL_HANDLE;
    }

private:
    static constexpr float TimelineFPS = 60.0f;
    static constexpr float FixedTimeStep = 1.0f / 60.0f;

    uint32_t computeTimelineFrameCount() const;
    uint32_t computeHistoryFrameCapacity() const { return computeTimelineFrameCount() + 1; }
    void failInitialization(const char* stage);
    bool rebuildGlobalThermalDomain();
    float getInitialTemperatureC(uint32_t runtimeModelId) const;
    void configureModelProperties(HeatModelRuntime& model, uint32_t runtimeModelId) const;
    bool rebuildDomainRuntime();
    void resetVoronoiTemperatures();

    void processResetTrigger();
    void forwardSim(float deltaTime);
    void releaseGmlsBuffers();
    void configureGMLSSurfaceWeights(bool heatVoronoiReady);
    bool recreateDescriptorPools();
    void configureModelSimResources();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& renderCommandPool;
    CommandPool& transferCommandPool;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    uint32_t maxFramesInFlight;

    std::unique_ptr<HeatSystemSurfaceStage> surfaceStage;

    HeatDomainRuntime domainRuntime;
    HeatPlaybackRuntime playbackRuntime;
    std::unique_ptr<HeatFluidRuntime> fluidRuntime;

    std::vector<std::vector<glm::vec3>> modelSurfacePositions;
    std::vector<std::vector<glm::vec3>> modelSurfaceNormals;
    std::vector<std::vector<uint32_t>> modelSurfaceTriangleIndices;
    std::vector<uint32_t> modelRuntimeModelIds;
    std::unordered_map<uint32_t, float> modelInitialTemperaturesCByRuntimeId;
    std::unordered_map<uint32_t, float> modelVolumetricPowerDensitiesByRuntimeId;
    std::unordered_map<uint32_t, bool> modelPrescribedTemperatureEnabledByRuntimeId;
    std::unordered_map<uint32_t, float> modelPrescribedTemperaturesCByRuntimeId;
    std::unordered_map<uint32_t, float> modelDensity;
    std::unordered_map<uint32_t, float> modelSpecificHeat;
    std::unordered_map<uint32_t, float> modelConductivity;
    units::LengthUnit worldUnit = units::defaultLengthUnit();

    TimelineSimulation timeline;
    float simulatedTime = 0.0f;
    bool shouldStepPhysics = false;
    bool physicsStepCompletedThisFrame = false;
    // Terminal solve failure: heat physics halts until the next reset
    bool physicsFailed = false;
    uint64_t completedSteps = 0;
    bool needsInitialCapture = false;
    float physicsAccumulator = 0.0f;
    bool temperatureBufferAIsCurrent = true;
    std::chrono::steady_clock::time_point lastUpdateTime = std::chrono::steady_clock::now();

    bool isActive = false;
    bool initialized = false;
    bool domainDirty = true;
    bool voronoiConfigDirty = true;
    bool heatParamsDirty = true;
    uint32_t processedResetCounter = 0;
    uint64_t boundInputConnectionId = 0;

    heat::DomainBoundaryConditions domainBoundaries = heat::createDefaultDomainBoundaries(HeatSimDefaults::ambientTemperatureC);

    std::unordered_map<uint32_t, uint32_t> simNodeCounts;
    std::unordered_map<uint32_t, std::array<float, 16>> modelLocalToWorldByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceStencilBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceStencilBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceWeightBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceWeightBufferOffsetByModelId;
    std::unordered_map<uint32_t, size_t> modelGMLSSurfaceWeightCountByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceGradientWeightBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsetByModelId;
    std::unordered_map<uint32_t, size_t> modelGMLSSurfaceGradientWeightCountByModelId;

    std::unique_ptr<HeatFluidVolumeStage> fluidVolumeStage;
    bool fluidVolumeDirty = false;
    bool lastFluidReplayFromHistory = false;
    uint32_t lastFluidDisplayFrame = 0;

    // Surface dispatches only run when displayed data changes, not every frame
    bool surfaceDataDirty = true;
    // Monotonic rendered frame counter for cadence decisions
    uint64_t renderedFrameCounter = 0;
};
