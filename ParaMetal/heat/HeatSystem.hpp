#pragma once

#include "heat/HeatContactRuntime.hpp"
#include "contact/ContactTypes.hpp"
#include "framegraph/ComputePass.hpp"
#include "util/Structs.hpp"
#include "HeatSystemSimRuntime.hpp"
#include "HeatSystemRuntime.hpp"
#include "HeatSystemPlayback.hpp"
#include "HeatSystemPresets.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

static constexpr int NUM_SUBSTEPS = 3;

class ModelRegistry;
class MemoryAllocator;
class VulkanDevice;
class CommandPool;
class HeatSystemSimStage;
class HeatSystemSurfaceStage;
class HeatSystemVoronoiStage;
class ContactSystemComputeStage;
class HeatSystemResources;
class HeatContactRuntime;

class HeatSystem : public ComputePass {
public:
    HeatSystem(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, ModelRegistry& resourceManager, HeatSystemResources& resources,
        uint32_t maxFramesInFlight, CommandPool& renderCommandPool);
    ~HeatSystem() override;

    void update() override;
    bool ensureConfigured();
    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;
    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);
    void cleanupResources();
    void cleanup();

    void setActive(bool active);
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
        const std::vector<uint32_t>& voronoiToSim);

    void setPlaybackState(bool paused, uint32_t resetCounter);
    void resetHeatState();
    void readbackTemperatures(uint32_t frameIndex);

    bool getIsActive() const { return isActive; }
    bool getIsPaused() const { return playbackControls.paused; }
    uint32_t getResetCounter() const { return playbackControls.resetCounter; }
    void setIsPaused(bool paused) { playbackControls.paused = paused; }
    bool isInitialized() const { return initialized; }
    bool hasDispatchableComputeWork() const override;
    bool voronoiReady() const;

    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return runtime.getActiveModels(); }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const { return runtime.getModelByRuntimeId(runtimeModelId); }

private:
    void failInitialization(const char* stage);
    bool rebuildRuntimeResources(bool forceDescriptorReallocate);
    bool rebuildVoronoiRuntime();
    bool initializeVoronoiMaterialNodes();
    void resetVoronoiTemperatures();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    HeatSystemResources& resources;
    CommandPool& renderCommandPool;

    HeatSystemRuntime runtime;
    HeatSystemSimRuntime simRuntime;
    std::vector<std::unique_ptr<HeatContactRuntime>> contactRuntimes;

    std::unique_ptr<HeatSystemSimStage> simStage;
    std::unique_ptr<HeatSystemSurfaceStage> surfaceStage;
    std::unique_ptr<HeatSystemVoronoiStage> voronoiStage;
    std::unique_ptr<ContactSystemComputeStage> contactStage;

    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<ContactCoupling> contactCouplings;
    std::unordered_map<uint32_t, float> initialTemps;
    float contactThermalConductance = 16000.0f;
    float simulationDuration = 5.0f;

    std::unordered_map<uint32_t, const voronoi::Node*> modelVoronoiNodesByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelVoronoiNodeBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelVoronoiNodeBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelSimNodeBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelSimNodeBufferOffsetByModelId;
    std::unordered_map<uint32_t, VkBuffer> modelSimGMLSInterfaceBufferByModelId;
    std::unordered_map<uint32_t, VkDeviceSize> modelSimGMLSInterfaceBufferOffsetByModelId;
    std::unordered_map<uint32_t, uint32_t> voronoiNodeCounts;
    std::unordered_map<uint32_t, uint32_t> simNodeCounts;
    std::unordered_map<uint32_t, uint32_t> simGMLSInterfaceCounts;
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

    uint32_t maxFramesInFlight;
    std::vector<VkCommandBuffer> computeCommandBuffers;

    bool isActive = false;
    bool initialized = false;
    bool voronoiConfigDirty = true;
    bool heatParamsDirty = true;
    bool contactCouplingsDirty = true;

    std::unordered_map<uint32_t, std::unique_ptr<HeatSystemPlayback>> playbacks;
    HeatSystemPlayback::Controls playbackControls;
    uint32_t processedResetCounter = 0;

    static constexpr uint32_t MAX_NODE_NEIGHBORS = 50;
};
