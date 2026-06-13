#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#include "HeatSystem.hpp"
#include "HeatSystemPresets.hpp"
#include "HeatSystemResources.hpp"
#include "framegraph/ComputePass.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

class VulkanDevice;
class MemoryAllocator;
class ModelRegistry;
class CommandPool;

class HeatSystemComputeController {
public:
    struct Config {
        bool active = false;
        bool paused = false;
        uint32_t resetCounter = 0;
        float contactThermalConductance = 16000.0f;
        float simulationDuration = 5.0f;
        std::vector<SupportingHalfedge::IntrinsicMesh> modelIntrinsicMeshes;
        std::vector<uint32_t> modelRuntimeModelIds;
        std::vector<VkBufferView> supportingHalfedgeViews;
        std::vector<VkBufferView> supportingAngleViews;
        std::vector<VkBufferView> halfedgeViews;
        std::vector<VkBufferView> edgeViews;
        std::vector<VkBufferView> triangleViews;
        std::vector<VkBufferView> lengthViews;
        std::vector<VkBufferView> inputHalfedgeViews;
        std::vector<VkBufferView> inputEdgeViews;
        std::vector<VkBufferView> inputTriangleViews;
        std::vector<VkBufferView> inputLengthViews;
        std::unordered_map<uint32_t, float> modelTemperatureByRuntimeId;
        std::unordered_map<uint32_t, uint32_t> modelBoundaryConditions;
        std::unordered_map<uint32_t, float> modelFixedTemperatureValues;
        std::unordered_map<uint32_t, float> modelDensity;
        std::unordered_map<uint32_t, float> modelSpecificHeat;
        std::unordered_map<uint32_t, float> modelConductivity;
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
        std::vector<ContactCoupling> contactCouplings;
        uint64_t computeHash = 0;
    };

    HeatSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        ModelRegistry& resourceManager,
        CommandPool& renderCommandPool,
        uint32_t maxFramesInFlight);

    bool isAnyHeatSystemActive() const;
    bool isAnyHeatSystemPaused() const;
    bool isHeatSystemActive(uint64_t socketKey) const;
    bool isHeatSystemPaused(uint64_t socketKey) const;

    void configure(uint64_t socketKey, const Config& config);
    void pushPlaybackState(uint64_t socketKey, bool paused, uint32_t resetCounter);
    void disable(uint64_t socketKey);
    void disableAll();
    std::vector<ComputePass*> getActiveSystems() const;
    const HeatSystem* getSystem(uint64_t socketKey) const;
    const Config* getConfig(uint64_t socketKey) const;

    void destroyHeatSystem(uint64_t socketKey);

private:
    struct SystemInstance {
        HeatSystemResources resources;
        std::unique_ptr<HeatSystem> system;
    };

    std::unique_ptr<HeatSystem> buildHeatSystem(HeatSystemResources& heatSystemResources);
    void configureHeatSystem(HeatSystem& system, const Config& config);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& renderCommandPool;

    std::unordered_map<uint64_t, std::unique_ptr<SystemInstance>> activeSystems;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    const uint32_t maxFramesInFlight;
};
