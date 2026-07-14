#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

#include "HeatSystem.hpp"
#include "HeatGpuStructs.hpp"
#include "HeatSystemPresets.hpp"
#include "framegraph/ComputePass.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

class VulkanDevice;
class MemoryAllocator;
class ModelRegistry;
class CommandPool;
class HeatModelRuntime;
struct HeatProduct;

class HeatSystemComputeController {
public:
    struct Config {
        bool active = false;
        bool paused = false;
        bool syntheticDirichletTestEnabled = false;
        uint32_t resetCounter = 0;
        uint32_t rewindFrame = heat::NoRewindFrame;
        float contactThermalConductance = 16000.0f;
        float simulationDuration = 5.0f;
        std::vector<std::vector<glm::vec3>> modelSurfacePositions;
        std::vector<std::vector<glm::vec3>> modelSurfaceNormals;
        std::vector<std::vector<uint32_t>> modelSurfaceTriangleIndices;
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
        std::unordered_map<uint32_t, float> modelInitialTemperaturesCByRuntimeId;
        std::unordered_map<uint32_t, uint32_t> modelBoundaryConditionTypesByRuntimeId;
        std::unordered_map<uint32_t, float> modelBoundaryTemperaturesCByRuntimeId;
        std::unordered_map<uint32_t, float> modelBoundaryHeatFluxesByRuntimeId;
        std::unordered_map<uint32_t, float> modelBoundaryHeatTransferCoefficientsByRuntimeId;
        std::unordered_map<uint32_t, float> modelVolumetricPowerDensitiesByRuntimeId;
        std::unordered_map<uint32_t, float> modelDensity;
        std::unordered_map<uint32_t, float> modelSpecificHeat;
        std::unordered_map<uint32_t, float> modelConductivity;
        std::unordered_map<uint32_t, VkBuffer> modelSimNodeBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelSimNodeBufferOffsetByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelSimNodeCouplingBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelSimNodeCouplingBufferOffsetByModelId;
        std::unordered_map<uint32_t, uint32_t> simNodeCounts;
        std::unordered_map<uint32_t, uint32_t> simNodeCouplingCounts;
        std::unordered_map<uint32_t, std::vector<glm::vec3>> modelNodePositionsByModelId;
        std::unordered_map<uint32_t, std::vector<voronoi::Node>> modelNodesByModelId;
        std::unordered_map<uint32_t, std::vector<voronoi::NodeCoupling>> modelNodeCouplingsByModelId;
        std::unordered_map<uint32_t, std::vector<uint32_t>> modelSurfaceNodeIdsByModelId;
        std::unordered_map<uint32_t, std::vector<float>> modelSurfacePatchAreasByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceStencilBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceStencilBufferOffsetByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceWeightBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceWeightBufferOffsetByModelId;
        std::unordered_map<uint32_t, size_t> modelGMLSSurfaceWeightCountByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceGradientWeightBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsetByModelId;
        std::unordered_map<uint32_t, size_t> modelGMLSSurfaceGradientWeightCountByModelId;
        std::vector<ContactCoupling> contactCouplings;
        uint64_t computeHash = 0;
        uint64_t structuralHash = 0;
    };

    HeatSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        ModelRegistry& resourceManager,
        CommandPool& renderCommandPool,
        CommandPool& transferCommandPool,
        uint32_t maxFramesInFlight);

    bool isAnyHeatSystemActive() const;
    bool isAnyHeatSystemPaused() const;

    void apply(uint64_t socketKey, const Config& config);
    bool buildProduct(uint64_t socketKey, HeatProduct& product);
    void remove(uint64_t socketKey);
    void disableAll();
    std::vector<ComputePass*> getActiveSystems() const;
    const HeatSystem* getSystem(uint64_t socketKey) const;
    const Config* getConfig(uint64_t socketKey) const;

private:
    std::unique_ptr<HeatSystem> buildHeatSystem();
    void configureHeatSystem(HeatSystem& system, const Config& config);
    void applyRuntimeState(HeatSystem& system, const Config& config);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    ModelRegistry& resourceManager;
    CommandPool& renderCommandPool;
    CommandPool& transferCommandPool;

    std::unordered_map<uint64_t, std::unique_ptr<HeatSystem>> systemsBySocket;
    std::unordered_map<uint64_t, Config> configuredConfigs;
    const uint32_t maxFramesInFlight;
};
