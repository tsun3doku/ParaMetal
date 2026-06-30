#pragma once

#include "heat/HeatContactRuntime.hpp"
#include "heat/TimelineSimulation.hpp"
#include "contact/ContactTypes.hpp"
#include "framegraph/ComputePass.hpp"
#include "util/Structs.hpp"
#include "HeatSystemSimRuntime.hpp"
#include "HeatSystemRuntime.hpp"
#include "HeatSystemPresets.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class ModelRegistry;
class MemoryAllocator;
class VulkanDevice;
class CommandPool;
class HeatSystemSimStage;
class HeatSystemSurfaceStage;
class HeatSystemDiffusionStage;
class ContactSystemComputeStage;
class HeatContactRuntime;

class HeatSystem : public ComputePass {
public:
    HeatSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, ModelRegistry& resourceManager,
        uint32_t maxFramesInFlight, CommandPool& renderCommandPool, CommandPool& transferCommandPool);
    ~HeatSystem() override;

    void update() override;
    bool ensureConfigured();
    bool setupDescriptors(const std::vector<VkBuffer>& surfaceBuffers, const std::vector<VkDeviceSize>& surfaceOffsets, const std::vector<VkBuffer>& gradientBuffers, const std::vector<VkDeviceSize>& gradientOffsets);
    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;
    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);
    void cleanupResources();
    void cleanup();

    void setActive(bool active);
    void setRewindFrame(uint32_t frame);
    void setPlaybackState(bool paused, uint32_t resetCounter);
    void setParams(float contactThermalConductance, float simulationDuration);
    void setContactCouplings(const std::vector<ContactCoupling>& contactCouplings);
    void setHeatModels(
        const std::vector<SupportingHalfedge::IntrinsicMesh>& modelIntrinsicMeshes,
        const std::vector<uint32_t>& modelRuntimeModelIds,
        const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId,
        const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditions,
        const std::unordered_map<uint32_t, float>& modelFixedTemperatureValues,
        const std::unordered_map<uint32_t, float>& modelDensity,
        const std::unordered_map<uint32_t, float>& modelSpecificHeat,
        const std::unordered_map<uint32_t, float>& modelConductivity);

    void clearVoronoiInputs();
    void addVoronoiModelInput(
        uint32_t runtimeModelId,
        const voronoi::Node* nodes,
        uint32_t voronoiNodeCount,
        VkBuffer voronoiNodeBuffer,
        VkDeviceSize voronoiNodeBufferOffset,
        uint32_t simNodeCount,
        VkBuffer simNodeBuffer,
        VkDeviceSize simNodeBufferOffset,
        VkDeviceSize simNodeBufferSize,
        VkBuffer simGMLSInterfaceBuffer,
        VkDeviceSize simGMLSInterfaceBufferOffset,
        uint32_t simGMLSInterfaceCount,
        VkBuffer gmlsSurfaceStencilBuffer,
        VkDeviceSize gmlsSurfaceStencilBufferOffset,
        VkBuffer gmlsSurfaceWeightBuffer,
        VkDeviceSize gmlsSurfaceWeightBufferOffset,
        size_t gmlsSurfaceWeightCount,
        VkBuffer gmlsSurfaceGradientWeightBuffer,
        VkDeviceSize gmlsSurfaceGradientWeightBufferOffset,
        size_t gmlsSurfaceGradientWeightCount,
        const std::vector<uint32_t>& seedFlags,
        const std::vector<glm::vec3>& seedPositions,
        const std::vector<float>& simNodeVolumes,
        const std::vector<uint32_t>& voronoiToSim);

    void resetSimulationState();

    bool getIsActive() const { return isActive; }
    bool getIsPaused() const { return timeline.isPaused(); }
    bool isInitialized() const { return initialized; }
    bool hasDispatchableComputeWork() const override;
    bool voronoiReady() const;

    uint32_t getResetCounter() const { return timeline.getResetCounter(); }
    float getSimulationTimeSeconds() const { return timeline.getCurrentPosition(); }
    uint32_t getRecordedTimelineFrames() const;
    uint32_t getTimelineFrameCount() const;
    float getSimulationDurationSeconds() const { return timeline.getDuration(); }
    uint32_t getRewindFrame() const;

    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return runtime.getActiveModels(); }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const { return runtime.getModelByRuntimeId(runtimeModelId); }

private:
    static constexpr float TimelineFPS = 60.0f;
    static constexpr int NumSubsteps = 3;

    uint32_t computeTimelineFrameCount() const;
    uint32_t computeHistoryFrameCapacity() const { return computeTimelineFrameCount() + 1; }
    void failInitialization(const char* stage);
    bool rebuildVoronoiRuntime();
    bool initializeVoronoiMaterialNodes();
    void resetVoronoiTemperatures();

    void processResetTrigger();
    void forwardSim(float deltaTime);
    void rebuildStencilKDTrees();
    void configureGMLSSurfaceWeights(bool heatVoronoiReady);
    bool recreateDescriptorPools();
    void configureModelSimResources();
    void rebuildContactRuntimes();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& renderCommandPool;
    CommandPool& transferCommandPool;
    std::vector<VkCommandBuffer> computeCommandBuffers;
    uint32_t maxFramesInFlight;

    std::unique_ptr<HeatSystemSimStage> simStage;
    std::unique_ptr<HeatSystemSurfaceStage> surfaceStage;
    std::unique_ptr<HeatSystemDiffusionStage> diffusionStage;
    std::unique_ptr<ContactSystemComputeStage> contactStage;

    HeatSystemRuntime runtime;
    HeatSystemSimRuntime simRuntime;
    std::vector<std::unique_ptr<HeatContactRuntime>> contactRuntimes;

    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<ContactCoupling> contactCouplings;
    std::unordered_map<uint32_t, float> initialTemps;
    float contactThermalConductance = 16000.0f;

    TimelineSimulation timeline;
    float simulatedTime = 0.0f;
    bool shouldStepPhysics = false;
    bool needsInitialCapture = false;

    bool isActive = false;
    bool initialized = false;
    bool voronoiConfigDirty = true;
    bool heatParamsDirty = true;
    bool contactCouplingsDirty = true;
    uint32_t processedResetCounter = 0;

    std::unordered_map<uint32_t, const voronoi::Node*> modelVoronoiNodesByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelVoronoiNodeBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelVoronoiNodeBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelSimNodeBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelSimNodeBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelSimNodeBufferSizeByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelSimGMLSInterfaceBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelSimGMLSInterfaceBufferOffsetByModelId;
    std::unordered_map<uint32_t, uint32_t> voronoiNodeCounts;
    std::unordered_map<uint32_t, uint32_t> simNodeCounts;
    std::unordered_map<uint32_t, uint32_t> simGMLSInterfaceCounts;
    std::unordered_map<uint32_t, std::vector<float>> modelSimNodeVolumesByModelId;
    std::unordered_map<uint32_t, std::vector<uint32_t>> modelVoronoiToSimByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceStencilBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceStencilBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceWeightBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceWeightBufferOffsetByModelId;
    std::unordered_map<uint32_t, size_t> modelGMLSSurfaceWeightCountByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceGradientWeightBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsetByModelId;
    std::unordered_map<uint32_t, size_t> modelGMLSSurfaceGradientWeightCountByModelId;
    std::unordered_map<uint32_t, std::vector<uint32_t>> modelVoronoiSeedFlagsByModelId;
    std::unordered_map<uint32_t, std::vector<glm::vec3>> modelVoronoiSeedPositionsByModelId;
};
