#pragma once

#include "heat/HeatContactRuntime.hpp"
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
    ~HeatSystem();

    void update() override;
    void ensureConfigured();
    void setActive(bool active);
    bool isInitialized() const { return initialized; }

    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;

    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);

    void cleanupResources();
    void cleanup();

    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }

    bool getIsActive() const { return isActive; }
    bool getIsPaused() const { return isPaused; }
    void setIsPaused(bool paused) { isPaused = paused; }
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return runtime.getActiveModels(); }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const { return runtime.getModelByRuntimeId(runtimeModelId); }
    bool hasDispatchableComputeWork() const override;
    bool voronoiReady() const;
    void resetHeatState(const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId);
    void setHeatModels(
        const std::vector<SupportingHalfedge::IntrinsicMesh>& modelIntrinsicMeshes,
        const std::vector<uint32_t>& modelRuntimeModelIds,
        const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId,
        const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditions,
        const std::unordered_map<uint32_t, float>& modelFixedTemperatureValues,
        const std::unordered_map<uint32_t, float>& modelDensity,
        const std::unordered_map<uint32_t, float>& modelSpecificHeat,
        const std::unordered_map<uint32_t, float>& modelConductivity);
    void setParams(float contactThermalConductance);
    void setContactCouplings(const std::vector<ContactCoupling>& contactCouplings);
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

    void readbackTemperatures(uint32_t frameIndex);

private:
    void failInitialization(const char* stage);
    bool rebuildRuntimeResources(bool forceDescriptorReallocate);
    bool rebuildVoronoiRuntime();
    bool initializeVoronoiMaterialNodes();
    void resetVoronoiTemperatures(const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    HeatSystemResources& resources;
    CommandPool& renderCommandPool;
    HeatSystemRuntime runtime;
    HeatSystemSimRuntime simRuntime;
    std::vector<std::unique_ptr<HeatContactRuntime>> contactRuntimes;

    std::vector<uint32_t> modelRuntimeModelIds;
    std::vector<ContactCoupling> contactCouplings;
    float contactThermalConductance = 16000.0f;
    bool contactCouplingsDirty = true;

    std::unique_ptr<HeatSystemSimStage> simStage;
    std::unique_ptr<HeatSystemSurfaceStage> surfaceStage;
    std::unique_ptr<HeatSystemVoronoiStage> voronoiStage;
    std::unique_ptr<ContactSystemComputeStage> contactStage;
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
    bool isPaused = false;
    bool initialized = false;
    bool voronoiConfigDirty = true;
    bool heatParamsDirty = true;
    static constexpr uint32_t MAX_NODE_NEIGHBORS = 50;
};

