#pragma once

#include "mesh/remesher/SupportingHalfedge.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

class CommandPool;
class HeatModelRuntime;
class MemoryAllocator;
class VulkanDevice;

class HeatSystemRuntime {
public:
    HeatSystemRuntime() = default;
    ~HeatSystemRuntime() = default;

    void setHeatModels(
        const std::vector<SupportingHalfedge::IntrinsicMesh>& modelIntrinsicMeshes,
        const std::vector<uint32_t>& modelRuntimeModelIds,
        const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId,
        const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditions,
        const std::unordered_map<uint32_t, float>& modelFixedTemperatureValues,
        const std::unordered_map<uint32_t, float>& modelDensity,
        const std::unordered_map<uint32_t, float>& modelSpecificHeat,
        const std::unordered_map<uint32_t, float>& modelConductivity);
    bool ensureModelBindings(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool);
    bool needsRebuild() const { return modelsDirty; }

    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return activeModels; }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const;

    void cleanup();

private:
    std::vector<SupportingHalfedge::IntrinsicMesh> activeModelIntrinsicMeshes;
    std::vector<uint32_t> activeModelRuntimeModelIds;
    std::unordered_map<uint32_t, float> activeModelTemperatureByRuntimeId;
    std::unordered_map<uint32_t, uint32_t> activeModelBoundaryConditions;
    std::unordered_map<uint32_t, float> activeModelFixedTemperatureValues;
    std::unordered_map<uint32_t, float> activeModelDensity;
    std::unordered_map<uint32_t, float> activeModelSpecificHeat;
    std::unordered_map<uint32_t, float> activeModelConductivity;
    std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>> activeModels;
    bool modelsDirty = true;
};
