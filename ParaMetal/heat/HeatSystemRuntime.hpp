#pragma once

#include <glm/vec3.hpp>

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
        const std::vector<std::vector<glm::vec3>>& modelSurfacePositions,
        const std::vector<std::vector<glm::vec3>>& modelSurfaceNormals,
        const std::vector<std::vector<uint32_t>>& modelSurfaceTriangleIndices,
        const std::vector<uint32_t>& modelRuntimeModelIds,
        const std::unordered_map<uint32_t, float>& modelInitialTemperaturesCByRuntimeId,
        const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditionTypesByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelBoundaryTemperaturesCByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelBoundaryHeatFluxesByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelBoundaryHeatTransferCoefficientsByRuntimeId,
        const std::unordered_map<uint32_t, float>& modelVolumetricPowerDensitiesByRuntimeId,
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
    float getInitialTemperatureC(uint32_t modelId) const;
    void configureModelProperties(HeatModelRuntime* model, uint32_t modelId) const;
    std::vector<std::vector<glm::vec3>> activeModelSurfacePositions;
    std::vector<std::vector<glm::vec3>> activeModelSurfaceNormals;
    std::vector<std::vector<uint32_t>> activeModelSurfaceTriangleIndices;
    std::vector<uint32_t> activeModelRuntimeModelIds;
    std::unordered_map<uint32_t, float> activeModelInitialTemperaturesCByRuntimeId;
    std::unordered_map<uint32_t, uint32_t> activeModelBoundaryConditionTypesByRuntimeId;
    std::unordered_map<uint32_t, float> activeModelBoundaryTemperaturesCByRuntimeId;
    std::unordered_map<uint32_t, float> activeModelBoundaryHeatFluxesByRuntimeId;
    std::unordered_map<uint32_t, float> activeModelBoundaryHeatTransferCoefficientsByRuntimeId;
    std::unordered_map<uint32_t, float> activeModelVolumetricPowerDensitiesByRuntimeId;
    std::unordered_map<uint32_t, float> activeModelDensity;
    std::unordered_map<uint32_t, float> activeModelSpecificHeat;
    std::unordered_map<uint32_t, float> activeModelConductivity;
    std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>> activeModels;
    bool modelsDirty = true;
};
