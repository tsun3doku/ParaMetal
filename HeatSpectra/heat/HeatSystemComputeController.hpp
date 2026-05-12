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
#include "runtime/RuntimeProducts.hpp"

class VulkanDevice;
class MemoryAllocator;
class ModelRegistry;
class CommandPool;

class HeatSystemComputeController {
public:
    struct Config {
        bool active = false;
        bool paused = false;
        bool resetRequested = false;
        float contactThermalConductance = 16000.0f;
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
        std::unordered_map<uint32_t, VkBuffer> modelGMLSInterfaceBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSInterfaceBufferOffsetByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelSeedFlagsBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelSeedFlagsBufferOffsetByModelId;
        std::unordered_map<uint32_t, uint32_t> modelVoronoiNodeCountByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceStencilBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceStencilBufferOffsetByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceWeightBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceWeightBufferOffsetByModelId;
        std::unordered_map<uint32_t, VkBuffer> modelGMLSSurfaceGradientWeightBufferByModelId;
        std::unordered_map<uint32_t, VkDeviceSize> modelGMLSSurfaceGradientWeightBufferOffsetByModelId;
        std::unordered_map<uint32_t, std::vector<uint32_t>> modelVoronoiSeedFlagsByModelId;
        std::unordered_map<uint32_t, std::vector<glm::vec3>> modelVoronoiSeedPositionsByModelId;
        std::vector<uint32_t> surfaceRuntimeModelIds;
        std::vector<VkBufferView> surfaceSupportingHalfedgeViews;
        std::vector<VkBufferView> surfaceSupportingAngleViews;
        std::vector<VkBufferView> surfaceHalfedgeViews;
        std::vector<VkBufferView> surfaceEdgeViews;
        std::vector<VkBufferView> surfaceTriangleViews;
        std::vector<VkBufferView> surfaceLengthViews;
        std::vector<VkBufferView> surfaceInputHalfedgeViews;
        std::vector<VkBufferView> surfaceInputEdgeViews;
        std::vector<VkBufferView> surfaceInputTriangleViews;
        std::vector<VkBufferView> surfaceInputLengthViews;
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
    void disable(uint64_t socketKey);
    void disableAll();
    std::vector<ComputePass*> getActiveSystems() const;
    bool exportProduct(uint64_t socketKey, HeatProduct& outProduct) const;

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

inline uint64_t buildComputeHash(const HeatSystemComputeController::Config& config) {
    uint64_t hash = 1469598103934665603ull;
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelIntrinsicMeshes.size()));
    for (const SupportingHalfedge::IntrinsicMesh& mesh : config.modelIntrinsicMeshes) {
        hash = RuntimeProductHash::mixPodVector(hash, mesh.vertices);
        hash = RuntimeProductHash::mixPodVector(hash, mesh.indices);
        hash = RuntimeProductHash::mixPodVector(hash, mesh.faceIds);
        hash = RuntimeProductHash::mixPodVector(hash, mesh.triangles);
    }
    hash = RuntimeProductHash::mixPodVector(hash, config.modelRuntimeModelIds);
    hash = RuntimeProductHash::mixPod(hash, config.contactThermalConductance);
    hash = RuntimeProductHash::mixPodVector(hash, config.supportingHalfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.supportingAngleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.halfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.edgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.triangleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.lengthViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.inputHalfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.inputEdgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.inputTriangleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.inputLengthViews);
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelTemperatureByRuntimeId.size()));
    for (const auto& [id, temp] : config.modelTemperatureByRuntimeId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, temp);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelBoundaryConditions.size()));
    for (const auto& [id, bc] : config.modelBoundaryConditions) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, static_cast<uint32_t>(bc));
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelFixedTemperatureValues.size()));
    for (const auto& [id, temp] : config.modelFixedTemperatureValues) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, temp);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelDensity.size()));
    for (const auto& [id, density] : config.modelDensity) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, density);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelSpecificHeat.size()));
    for (const auto& [id, specificHeat] : config.modelSpecificHeat) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, specificHeat);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelConductivity.size()));
    for (const auto& [id, conductivity] : config.modelConductivity) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, conductivity);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelVoronoiNodeBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelVoronoiNodeBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelVoronoiNodeBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelVoronoiNodeBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSInterfaceBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelGMLSInterfaceBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSInterfaceBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelGMLSInterfaceBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelSeedFlagsBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelSeedFlagsBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelSeedFlagsBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelSeedFlagsBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelVoronoiNodeCountByModelId.size()));
    for (const auto& [id, count] : config.modelVoronoiNodeCountByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, count);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceStencilBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelGMLSSurfaceStencilBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceStencilBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelGMLSSurfaceStencilBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceWeightBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelGMLSSurfaceWeightBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceWeightBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelGMLSSurfaceWeightBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceGradientWeightBufferByModelId.size()));
    for (const auto& [id, buffer] : config.modelGMLSSurfaceGradientWeightBufferByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, buffer);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId.size()));
    for (const auto& [id, offset] : config.modelGMLSSurfaceGradientWeightBufferOffsetByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPod(hash, offset);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelVoronoiSeedFlagsByModelId.size()));
    for (const auto& [id, flags] : config.modelVoronoiSeedFlagsByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPodVector(hash, flags);
    }
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.modelVoronoiSeedPositionsByModelId.size()));
    for (const auto& [id, positions] : config.modelVoronoiSeedPositionsByModelId) {
        hash = RuntimeProductHash::mixPod(hash, id);
        hash = RuntimeProductHash::mixPodVector(hash, positions);
    }
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceRuntimeModelIds);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceSupportingHalfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceSupportingAngleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceHalfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceEdgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceTriangleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceLengthViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceInputHalfedgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceInputEdgeViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceInputTriangleViews);
    hash = RuntimeProductHash::mixPodVector(hash, config.surfaceInputLengthViews);
    hash = RuntimeProductHash::mix(hash, static_cast<uint64_t>(config.contactCouplings.size()));
    for (const ContactCoupling& coupling : config.contactCouplings) {
        hash = RuntimeProductHash::mixPod(hash, coupling.modelARuntimeModelId);
        hash = RuntimeProductHash::mixPod(hash, coupling.modelBRuntimeModelId);
        hash = RuntimeProductHash::mixPodVector(hash, coupling.modelBTriangleIndices);
        hash = RuntimeProductHash::mixPod(hash, coupling.contactPairCount);
    }
    return hash;
}
